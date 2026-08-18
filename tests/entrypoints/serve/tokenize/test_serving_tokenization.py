# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from jsonschema import Draft202012Validator
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
from vllm.exceptions import VLLMValidationError
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

    assert isinstance(request, TokenizeResponsesRequest)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {"input": "responses", "messages": []},
            TokenizeChatRequest,
        ),
        (
            {"prompt": "completion", "messages": []},
            TokenizeCompletionRequest,
        ),
        (
            {"prompt": "completion", "input": "responses"},
            TokenizeCompletionRequest,
        ),
        (
            {
                "prompt": "completion",
                "messages": [],
                "input": "responses",
            },
            TokenizeCompletionRequest,
        ),
        ({}, None),
        (
            {"input": "responses", "prompt": None},
            TokenizeResponsesRequest,
        ),
        (
            {"messages": [], "prompt": None},
            TokenizeChatRequest,
        ),
        (
            {"input": "responses", "messages": "invalid"},
            None,
        ),
        ({"input": "responses", "messages": None}, TokenizeResponsesRequest),
        ({"input": "responses", "prompt": {"id": "resp_123"}}, None),
    ],
)
def test_tokenize_request_schema_matches_runtime_routing(payload, expected_type):
    adapter = TypeAdapter(TokenizeRequest)
    schema_validator = Draft202012Validator(adapter.json_schema())

    assert schema_validator.is_valid(payload) is (expected_type is not None)

    if expected_type is None:
        with pytest.raises((ValidationError, VLLMValidationError)):
            adapter.validate_python(payload)
    else:
        request = adapter.validate_python(payload)
        assert isinstance(request, expected_type)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (
            {"model": MODEL_NAME, "input": "responses", "prompt": "completion"},
            TokenizeCompletionRequest,
        ),
        (
            {"model": MODEL_NAME, "input": "responses", "messages": []},
            TokenizeChatRequest,
        ),
        (
            {"model": MODEL_NAME, "prompt": "completion", "messages": []},
            TokenizeCompletionRequest,
        ),
        (
            {"model": MODEL_NAME, "input": "responses"},
            TokenizeResponsesRequest,
        ),
    ],
)
def test_tokenize_request_preserves_legacy_routing(payload, expected_type):
    request = TypeAdapter(TokenizeRequest).validate_python(payload)

    assert isinstance(request, expected_type)


def test_tokenize_request_requires_a_request_shape():
    with pytest.raises(ValidationError):
        TypeAdapter(TokenizeRequest).validate_python({"model": MODEL_NAME})


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


def _build_serving_tokenization(engine: AsyncLLM) -> ServingTokenization:
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
    )


def _build_mock_engine() -> MagicMock:
    mock_engine = MagicMock(spec=AsyncLLM)
    mock_engine.errored = False
    mock_engine.model_config = MockModelConfig()
    mock_engine.input_processor = MagicMock()
    mock_engine.renderer = MagicMock()
    return mock_engine


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
    mock_engine = _build_mock_engine()

    serving = _build_serving_tokenization(mock_engine)
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
        tool_server=None,
        skip_mm_cache=True,
    )


@pytest.mark.asyncio
async def test_tokenize_responses_rejects_server_managed_harmony_tools():
    mock_engine = _build_mock_engine()
    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.use_harmony = True
    serving.online_renderer.render_responses = AsyncMock()
    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Search for vLLM",
        tools=[{"type": "web_search_preview"}],
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert response.error.param == "tools"
    serving.online_renderer.render_responses.assert_not_awaited()


@pytest.mark.asyncio
async def test_tokenize_responses_allows_user_supplied_harmony_tools():
    mock_engine = _build_mock_engine()
    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.use_harmony = True
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(engine_input={"prompt_token_ids": [7, 8, 9]})
    )
    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Look up the weather",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [7, 8, 9]


@pytest.mark.asyncio
async def test_tokenize_responses_rejects_empty_rendered_token_ids():
    mock_engine = _build_mock_engine()
    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(engine_input={"prompt_token_ids": []})
    )
    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        max_output_tokens=100,
        truncation="auto",
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert response.error.message == "No token_ids rendered"


@pytest.mark.asyncio
async def test_tokenize_responses_returns_renderer_error_response_unchanged():
    mock_engine = _build_mock_engine()
    serving = _build_serving_tokenization(mock_engine)
    renderer_error = ErrorResponse(
        error={
            "message": "renderer rejected the request",
            "type": "invalid_request_error",
            "code": 400,
        }
    )
    serving.online_renderer.render_responses = AsyncMock(return_value=renderer_error)

    request = TokenizeResponsesRequest(model=MODEL_NAME, input="Test prompt")
    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response is renderer_error


@pytest.mark.asyncio
async def test_tokenize_responses_returns_token_strings_for_rendered_token_ids():
    mock_engine = _build_mock_engine()
    tokenizer = mock_engine.renderer.get_tokenizer.return_value
    tokenizer.convert_ids_to_tokens.return_value = [
        "seven",
        "eight",
        "nine",
    ]
    serving = _build_serving_tokenization(mock_engine)
    serving.online_renderer.render_responses = AsyncMock(
        return_value=MagicMock(engine_input={"prompt_token_ids": [7, 8, 9]})
    )

    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        return_token_strs=True,
    )
    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert response.tokens == [7, 8, 9]
    assert response.token_strs == ["seven", "eight", "nine"]
    tokenizer.convert_ids_to_tokens.assert_called_once_with([7, 8, 9])


@pytest.mark.asyncio
async def test_tokenize_responses_rejects_previous_response_id():
    mock_engine = _build_mock_engine()

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


@pytest.mark.asyncio
async def test_tokenize_responses_rejects_untrusted_request_template():
    mock_engine = _build_mock_engine()

    serving = _build_serving_tokenization(mock_engine)
    serving.trust_request_chat_template = False
    serving.online_renderer.render_responses = AsyncMock()
    request = TokenizeResponsesRequest(
        model=MODEL_NAME,
        input="Test prompt",
        chat_template_kwargs={"chat_template": "{{ messages }}"},
    )

    response = await serving.create_tokenize(request, MagicMock(headers={}))

    assert isinstance(response, ErrorResponse)
    assert response.error.code == 400
    assert "untrusted chat template" in response.error.message.lower()
    serving.online_renderer.render_responses.assert_not_awaited()
