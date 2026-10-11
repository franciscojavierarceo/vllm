# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.decisions.question_types import (
    Question,
    StructuredDecisionError,
    build_question,
)
from vllm.entrypoints.openai.decisions.serving import BaseServingDecisions
from vllm.entrypoints.openai.decisions.strategies import DecisionLimits
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from .protocol import (
    DecisionUsage,
    QuestionDiagnostics,
    StructuredDecisionRequest,
    StructuredDecisionResponse,
)


def state_text(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def parse_questions(
    request: StructuredDecisionRequest,
    limits: DecisionLimits,
) -> list[Question]:
    if not request.questions:
        raise StructuredDecisionError("questions: needs at least one question")
    if len(request.questions) > limits.max_questions:
        raise StructuredDecisionError(
            f"questions: at most {limits.max_questions} for this model"
        )
    questions = []
    for qid, spec in request.questions.items():
        if spec.model_extra:
            raise StructuredDecisionError(
                f"question {qid!r}: unknown field(s) {sorted(spec.model_extra)}"
            )
        questions.append(
            build_question(
                qid,
                spec.type,
                spec.instructions,
                spec.criteria,
                limits.max_options,
            )
        )
    return questions


class ServingStructuredDecisions(BaseServingDecisions):
    async def create_decision(
        self,
        request: StructuredDecisionRequest,
        raw_request: Request | None = None,
    ) -> StructuredDecisionResponse | ErrorResponse:
        result = await self._read(
            request,
            raw_request,
            lambda: parse_questions(request, self.limits),
            state_text(request.state),
            instructions=request.instructions,
            chat_template_kwargs=request.chat_template_kwargs,
            priority=request.priority,
            cache_salt=request.cache_salt,
            default_request_id=request.request_id,
        )
        if isinstance(result, ErrorResponse):
            return result

        answers: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, QuestionDiagnostics] = {}
        for q, read in zip(result.questions, result.reads):
            answers[q.id] = q.type.answer(q, read.probs, read.label_mass)
            diagnostics[q.id] = QuestionDiagnostics(
                label_mass=read.label_mass, argmax_is_label=read.argmax_is_label
            )
        return StructuredDecisionResponse(
            id=result.request_id,
            model=result.model_name,
            answers=answers,
            usage=DecisionUsage(
                input_tokens=sum(r.input_tokens for r in result.reads),
                output_tokens=sum(r.output_tokens for r in result.reads),
            ),
            diagnostics=diagnostics,
        )
