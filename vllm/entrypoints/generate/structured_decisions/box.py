# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-image grounding for System One Box questions."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vllm.entrypoints.openai.decisions.question_types import StructuredDecisionError
from vllm.entrypoints.openai.decisions.strategies import (
    NextTokenStrategy,
    ReadPromptRequest,
    ReadStrategy,
)
from vllm.exceptions import GenerationError
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.utils.async_utils import merge_async_iterators

MAX_BOX_TOKENS = 128


@dataclass(frozen=True)
class BoxQuestion:
    id: str
    instructions: str


@dataclass(frozen=True)
class BoxRead:
    answer: dict[str, Any]
    input_tokens: int
    output_tokens: int


class BoxPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bbox_2d: list[Annotated[int, Field(strict=True, ge=0, le=1000)]] = Field(
        min_length=4, max_length=4
    )

    @model_validator(mode="after")
    def check_edges(self) -> "BoxPrediction":
        left, top, right, bottom = self.bbox_2d
        if left >= right or top >= bottom:
            raise ValueError("Box edges must enclose a positive area")
        return self


def box_strategy(strategy: ReadStrategy, images: list[str]) -> NextTokenStrategy:
    config = strategy.context.engine_client.model_config
    if (
        type(strategy) is not NextTokenStrategy
        or config.architecture
        not in {
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        }
        or not config.is_multimodal_model
    ):
        raise StructuredDecisionError(
            "Box questions require a Qwen3.5 vision generation model"
        )
    if len(images) != 1:
        raise StructuredDecisionError("Box questions require exactly one image")
    if config.logprobs_mode != "raw_logprobs":
        raise StructuredDecisionError("Box questions require raw_logprobs")
    return strategy


async def render_boxes(
    strategy: NextTokenStrategy,
    questions: list[BoxQuestion],
    state: list[dict[str, Any]],
    instructions: str | None,
    read_request: ReadPromptRequest,
) -> list[EngineInput]:
    inputs = []
    for question in questions:
        messages = [
            {**message, "content": list(message["content"])} for message in state
        ]
        messages[-1]["content"].append(
            {
                "type": "text",
                "text": (
                    f"\n\nLocate this target in the image: {question.instructions}\n"
                    "Return one tight bounding box as JSON: "
                    '{"bbox_2d": [left, top, right, bottom]}. '
                    "Use integer coordinates from 0 to 1000, measured from the "
                    "top-left of the full image. Return only the JSON object."
                ),
            }
        )
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})
        engine_input, prompt_ids = await strategy._render(read_request, messages)
        if (
            len(prompt_ids) + MAX_BOX_TOKENS
            > strategy.context.engine_client.model_config.max_model_len
        ):
            raise StructuredDecisionError(
                f"Box question {question.id!r}: prompt must leave "
                f"{MAX_BOX_TOKENS} tokens for coordinates"
            )
        inputs.append(engine_input)
    return inputs


async def read_boxes(
    strategy: NextTokenStrategy,
    questions: list[BoxQuestion],
    inputs: list[EngineInput],
    *,
    request_id: str,
    lora_request: LoRARequest | None,
    priority: int,
    trace_headers: Mapping[str, str] | None,
) -> list[BoxRead]:
    params = SamplingParams(
        max_tokens=MAX_BOX_TOKENS,
        temperature=0.0,
        logprobs=0,
        output_kind=RequestOutputKind.FINAL_ONLY,
        structured_outputs=StructuredOutputsParams(
            json=BoxPrediction.model_json_schema(), disable_any_whitespace=True
        ),
    )
    generators = [
        strategy.context.engine_client.generate(
            engine_input,
            params,
            f"{request_id}-box-{index}",
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )
        for index, engine_input in enumerate(inputs)
    ]
    results: list[RequestOutput | None] = [None] * len(questions)
    async for index, partial in merge_async_iterators(*generators):
        results[index] = partial
    reads = []
    for question, result in zip(questions, results):
        if result is None or not result.outputs:
            raise GenerationError(f"No Box output for question {question.id!r}")
        output = result.outputs[0]
        if output.finish_reason in {"error", "abort"}:
            raise GenerationError(f"Box generation failed for question {question.id!r}")
        if output.finish_reason == "length":
            raise GenerationError(
                f"Box generation exceeded its token budget for question {question.id!r}"
            )
        try:
            prediction = BoxPrediction.model_validate_json(output.text)
        except ValidationError as exc:
            raise GenerationError(
                f"Invalid Box coordinates for question {question.id!r}"
            ) from exc
        logprob = output.cumulative_logprob
        if (
            not output.token_ids
            or logprob is None
            or not math.isfinite(logprob)
            or logprob > 0
        ):
            raise GenerationError(
                f"Missing Box likelihood for question {question.id!r}"
            )
        reads.append(
            BoxRead(
                answer={
                    "type": "box",
                    "box": {
                        edge: value / 1000
                        for edge, value in zip(
                            ("left", "top", "right", "bottom"), prediction.bbox_2d
                        )
                    },
                    "confidence": math.exp(logprob / len(output.token_ids)),
                },
                input_tokens=len(result.prompt_token_ids or ()),
                output_tokens=len(output.token_ids),
            )
        )
    return reads
