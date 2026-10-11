# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import Request

from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.tracing import (
    contains_trace_headers,
    extract_trace_headers,
    log_tracing_disabled_warning,
)

from .adapters import input_text, make_answer, make_read_question
from .protocol import (
    DecisionRequest,
    DecisionResponse,
    InputTokensDetails,
    OpenAIDecisionUsage,
    OutputTokensDetails,
)
from .question_types import Question, StructuredDecisionError
from .strategies import QuestionRead, ReadStrategy


@dataclass
class DecisionReads:
    request_id: str
    model_name: str
    questions: list[Question]
    reads: list[QuestionRead]


class BaseServingDecisions(BaseServing):
    def __init__(
        self,
        models: OpenAIServingModels,
        strategy: ReadStrategy,
        *,
        request_logger: RequestLogger | None = None,
    ) -> None:
        super().__init__(
            models=models,
            model_config=strategy.context.engine_client.model_config,
            request_logger=request_logger,
        )
        self.strategy = strategy
        self.limits = strategy.limits()

    async def _get_trace_headers(
        self, headers: Mapping[str, str]
    ) -> Mapping[str, str] | None:
        if not contains_trace_headers(headers):
            return None
        if not await self.strategy.context.engine_client.is_tracing_enabled():
            log_tracing_disabled_warning()
            return None
        return extract_trace_headers(headers)

    async def _read(
        self,
        request: Any,
        raw_request: Request | None,
        parse_questions: Callable[[], list[Question]],
        state: str,
        *,
        instructions: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        priority: int = 0,
        cache_salt: str | None = None,
        default_request_id: str | None = None,
    ) -> DecisionReads | ErrorResponse:
        """Validate, admit and read ``parse_questions()`` about ``state``.
        Shared by /v1/decisions and /v1/systemone."""
        if (error := await self._check_model(request)) is not None:
            return error
        engine_client = self.strategy.context.engine_client
        if engine_client.errored:
            raise engine_client.dead_error

        base_id = self._base_request_id(raw_request, default=default_request_id)
        request_id = f"decision-{base_id}"
        try:
            questions = parse_questions()
            lora_request = self._maybe_get_adapters(request)
            engine_client.check_admission(len(questions))
            self._log_inputs(request_id, state, None, lora_request)
            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            reads = await self.strategy.read(
                questions,
                instructions,
                state,
                request_id=request_id,
                chat_template_kwargs=chat_template_kwargs,
                lora_request=lora_request,
                priority=priority,
                cache_salt=cache_salt,
                trace_headers=trace_headers,
            )
        except StructuredDecisionError as e:
            return self.create_error_response(e)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        return DecisionReads(
            request_id=request_id,
            model_name=self.models.model_name(lora_request),
            questions=questions,
            reads=reads,
        )


class OpenAIServingDecisions(BaseServingDecisions):
    def _parse_questions(self, request: DecisionRequest) -> list[Question]:
        if len(request.questions) > self.limits.max_questions:
            raise StructuredDecisionError(
                f"questions: at most {self.limits.max_questions} for this model"
            )
        return [
            make_read_question(i, question, self.limits.max_options)
            for i, question in enumerate(request.questions)
        ]

    async def create_decisions(
        self, request: DecisionRequest, raw_request: Request | None = None
    ) -> DecisionResponse | ErrorResponse:
        result = await self._read(
            request,
            raw_request,
            lambda: self._parse_questions(request),
            input_text(request.input),
        )
        if isinstance(result, ErrorResponse):
            return result
        reads = result.reads
        input_tokens = sum(read.input_tokens for read in reads)
        output_tokens = sum(read.output_tokens for read in reads)
        return DecisionResponse(
            model=result.model_name,
            answers=[
                make_answer(question, read_question, read.probs, read.label_mass)
                for question, read_question, read in zip(
                    request.questions, result.questions, reads
                )
            ],
            usage=OpenAIDecisionUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=sum(read.cached_tokens for read in reads),
                    cache_write_tokens=sum(read.cache_write_tokens for read in reads),
                ),
                output_tokens_details=OutputTokensDetails(),
            ),
        )
