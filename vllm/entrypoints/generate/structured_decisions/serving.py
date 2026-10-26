# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.decisions.question_types import (
    Question,
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.openai.decisions.serving import BaseServingDecisions
from vllm.entrypoints.openai.decisions.strategies import (
    D3Strategy,
    DecisionLimits,
    NextTokenStrategy,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.inputs import EngineInput
from vllm.multimodal.utils import encode_image_url

from .box import BoxQuestion, box_strategy, read_boxes, render_boxes
from .d3_box import GRID_READS, MAX_BOX_QUESTIONS, read_d3_boxes
from .protocol import (
    DecisionUsage,
    QuestionDiagnostics,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)


def parse_questions(
    request: StructuredDecisionRequest,
    limits: DecisionLimits,
) -> list[Question | BoxQuestion]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > limits.max_questions:
        raise StructuredDecisionError(
            f"questions: at most {limits.max_questions} for this model"
        )
    questions: list[Question | BoxQuestion] = []
    for qid, spec in request.questions.items():
        if spec.model_extra:
            raise StructuredDecisionError(
                f"question {qid!r}: unknown field(s) {sorted(spec.model_extra)}"
            )
        if spec.type == "box":
            if (
                not qid
                or not isinstance(spec.instructions, str)
                or not spec.instructions.strip()
            ):
                raise StructuredDecisionError(
                    "Box questions need a nonempty id and target instructions"
                )
            if spec.criteria is not None:
                raise StructuredDecisionError("Box questions do not take criteria")
            questions.append(BoxQuestion(qid, spec.instructions))
            continue
        questions.append(
            build_question(
                qid,
                spec.type,
                spec.instructions,
                spec.criteria,
                limits.max_options,
                labeled=limits.labeled,
            )
        )
    return questions


class ServingStructuredDecisions(BaseServingDecisions):
    def _media(self, request: StructuredDecisionRequest) -> dict[str, list[str]]:
        urls = {
            "image": request.images,
            "audio": request.audio,
            "video": request.videos,
        }
        media = {modality: items for modality, items in urls.items() if items}
        if unsupported := media.keys() - self.strategy.modalities:
            names = ", ".join(sorted(unsupported))
            raise StructuredDecisionError(f"this model does not take {names} inputs")
        return media

    async def create_decision(
        self,
        request: StructuredDecisionRequest,
        raw_request: Request | None = None,
    ) -> StructuredDecisionResponse | ErrorResponse:
        if (error := await self._check_model(request)) is not None:  # type: ignore[arg-type]
            return error
        engine_client = self.strategy.context.engine_client
        if engine_client.errored:
            raise engine_client.dead_error

        base_id = self._base_request_id(raw_request, default=request.request_id)
        request_id = f"decision-{base_id}"
        try:
            questions = parse_questions(request, self.limits)
            media = self._media(request)
            labels = [q for q in questions if isinstance(q, Question)]
            boxes = [q for q in questions if isinstance(q, BoxQuestion)]
            grounding = (
                box_strategy(self.strategy, request.images or []) if boxes else None
            )
            if isinstance(grounding, D3Strategy):
                if len(boxes) > MAX_BOX_QUESTIONS:
                    raise StructuredDecisionError(
                        f"D3 takes at most {MAX_BOX_QUESTIONS} Box questions "
                        "per request"
                    )
                if request.instructions:
                    raise StructuredDecisionError(
                        "this model reads a fixed prompt, so it takes no instructions"
                    )
                if request.videos:
                    raise StructuredDecisionError("D3 Box questions do not take videos")
            box_request = (
                grounding._read_request(
                    request.chat_template_kwargs, request.cache_salt
                )
                if isinstance(grounding, NextTokenStrategy)
                else None
            )
            lora_request = self._maybe_get_adapters(request)  # type: ignore[arg-type]
            label_requests = self.strategy.num_requests(labels) if labels else 0
            engine_client.check_admission(
                max(label_requests, GRID_READS)
                if isinstance(grounding, D3Strategy)
                else label_requests + len(boxes)
            )
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            state: str | list[dict[str, Any]] = self.strategy.render_state(
                request.state
            )
            box_inputs: list[EngineInput] = []
            if isinstance(grounding, NextTokenStrategy):
                assert box_request is not None
                images = await grounding._image_parts(request.images or [])
                state = [
                    {
                        "role": "user",
                        "content": [*images, {"type": "text", "text": state}],
                    }
                ]
                box_inputs = await render_boxes(
                    grounding, boxes, state, request.instructions, box_request
                )
                media = {}
            d3_image = None
            if isinstance(grounding, D3Strategy):
                images = await grounding._image_parts(request.images or [])
                d3_image = images[0]["image_pil"]
                if labels:
                    media = {
                        "image": [await asyncio.to_thread(encode_image_url, d3_image)]
                    }
            reads = (
                await self.strategy.read(
                    labels,
                    request.instructions,
                    state,
                    request_id=request_id,
                    chat_template_kwargs=request.chat_template_kwargs,
                    lora_request=lora_request,
                    priority=request.priority,
                    cache_salt=request.cache_salt,
                    trace_headers=trace_headers,
                    media=media,
                )
                if labels
                else []
            )
            box_reads = []
            if isinstance(grounding, D3Strategy):
                assert d3_image is not None and isinstance(state, str)
                box_reads = await read_d3_boxes(
                    grounding,
                    boxes,
                    d3_image,
                    state,
                    request_id=request_id,
                    chat_template_kwargs=request.chat_template_kwargs,
                    lora_request=lora_request,
                    priority=request.priority,
                    cache_salt=request.cache_salt,
                    trace_headers=trace_headers,
                )
            elif isinstance(grounding, NextTokenStrategy):
                box_reads = await read_boxes(
                    grounding,
                    boxes,
                    box_inputs,
                    request_id=request_id,
                    lora_request=lora_request,
                    priority=request.priority,
                    trace_headers=trace_headers,
                )
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        for q, read in zip(labels, reads):
            answers[q.id] = q.type.answer(q, read.probs, read.label_mass)
            if read.confidence is not None:
                answers[q.id]["confidence"] = read.confidence
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
        answers.update((q.id, read.answer) for q, read in zip(boxes, box_reads))
        return StructuredDecisionResponse(
            id=request_id,
            model=self.models.model_name(lora_request),
            answers={q.id: answers[q.id] for q in questions},
            usage=DecisionUsage(
                input_tokens=sum(r.input_tokens for r in reads)
                + sum(r.input_tokens for r in box_reads),
                output_tokens=sum(r.output_tokens for r in reads)
                + sum(r.output_tokens for r in box_reads),
            ),
            diagnostics=diagnostics,
        )
