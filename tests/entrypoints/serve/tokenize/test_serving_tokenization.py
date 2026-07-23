# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm.config.multimodal import MultiModalConfig
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.tokenize.protocol import (
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeRequest,
    TokenizeResponsesRequest,
)
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.async_llm import AsyncLLM

MODEL_NAME = "openai-community/gpt2"
BASE_MODEL_PATHS = [
    BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME),
]

pytestmark = pytest.mark.skip_global_cleanup


def test_tokenize_request_accepts_responses_input():
    request = TypeAdapter(TokenizeRequest).validate_python(
        {
            "model": MODEL_NAME,
            "input": "Test prompt",
            "instructions": "Be brief.",
        }
    )

    assert type(request).__name__ == "TokenizeResponsesRequest"


def test_tokenize_request_routes_responses_prompt_object_to_responses_validation():
    with pytest.raises(ValidationError, match="prompt template is not supported"):
        TypeAdapter(TokenizeRequest).validate_python(
            {
                "model": MODEL_NAME,
                "input": "Test prompt",
                "prompt": {"id": "pmpt_123", "version": "1"},
            }
        )


def test_tokenize_request_schema_uses_one_of():
    schema = TypeAdapter(TokenizeRequest).json_schema()

    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"model": MODEL_NAME},
        {"model": MODEL_NAME, "input": "responses", "messages": []},
        {"model": MODEL_NAME, "prompt": "completion", "messages": []},
    ],
)
def test_tokenize_request_requires_exactly_one_request_shape(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(TokenizeRequest).validate_python(payload)


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    runner_type = "generate"
    model = MODEL_NAME
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    hf_text_config = MockHFConfig()
    logits_processors: list[str] | None = None
    diff_sampling_param: dict | None = None
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    encoder_config = None
    generation_config: str = "auto"
    media_io_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    skip_tokenizer_init = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


def _build_serving_tokenization(
    engine: AsyncLLM,
    *,
    tool_server=None,
) -> ServingTokenization:
    models = OpenAIServingModels(
        engine_client=engine,
        base_model_paths=BASE_MODEL_PATHS,
    )
    online_renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    return ServingTokenization(
        models,
        online_renderer=online_renderer,
        chat_template=None,
        chat_template_content_format="auto",
        tool_server=tool_server,
    )


@pytest.mark.asyncio
async def test_tokenize_chat_skips_mm_cache_for_renderer_only_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.preprocess_chat = AsyncMock(
        return_value=(
            [{"role": "user", "content": "Test"}],
            [{"prompt_token_ids": [1, 2, 3]}],
        )
    )

    request = TokenizeChatRequest(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Test prompt"}],
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.online_renderer.preprocess_chat.call_args.kwargs["skip_mm_cache"]
        is True
    )


@pytest.mark.asyncio
async def test_tokenize_completion_skips_mm_cache_for_renderer_only_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.preprocess_completion = AsyncMock(
        return_value=[{"prompt_token_ids": [1, 2, 3]}]
    )

    request = TokenizeCompletionRequest(
        model=MODEL_NAME,
        prompt="Test prompt",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [1, 2, 3]
    assert (
        serving.online_renderer.preprocess_completion.call_args.kwargs["skip_mm_cache"]
        is True
    )


@pytest.mark.asyncio
async def test_tokenize_responses_uses_stateless_online_renderer_path():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    tool_server = MagicMock()
    serving = _build_serving_tokenization(mock_engine, tool_server=tool_server)
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(
            messages=[{"role": "user", "content": "Test prompt"}],
            engine_input={"prompt_token_ids": [7, 8, 9]},
        )
    )

    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        instructions="Be brief.",
    )
    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [7, 8, 9]
    serving.online_renderer.render_responses.assert_awaited_once_with(
        request,
        previous_messages=None,
        previous_response_outputs=None,
        tool_server=tool_server,
        skip_mm_cache=True,
    )


@pytest.mark.asyncio
async def test_tokenize_responses_rejects_previous_response_id():
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()

    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.render_responses = AsyncMock()
    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        previous_response_id="resp_previous",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert response.error.param == "previous_response_id"
    serving.online_renderer.render_responses.assert_not_awaited()
