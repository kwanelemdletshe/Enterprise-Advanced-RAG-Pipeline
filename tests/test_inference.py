import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import inference


def test_gpt5_chat_completion_omits_unsupported_temperature(monkeypatch):
    monkeypatch.setenv("AZURE_INFERENCE_ENDPOINT", "https://example.services.ai.azure.com")
    monkeypatch.setattr(inference, "get_azure_inference_token", lambda _: "token")
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "READY"}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(inference.requests, "post", post)

    result = inference.chat_completion(
        credential=Mock(),
        model="gpt-5-mini",
        prompt="Reply with READY.",
        temperature=0.0,
    )

    assert result == "READY"
    assert "temperature" not in post.call_args.kwargs["json"]


def test_other_chat_models_keep_configured_temperature(monkeypatch):
    monkeypatch.setenv("AZURE_INFERENCE_ENDPOINT", "https://example.services.ai.azure.com")
    monkeypatch.setattr(inference, "get_azure_inference_token", lambda _: "token")
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "READY"}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(inference.requests, "post", post)

    inference.chat_completion(
        credential=Mock(),
        model="gpt-4o",
        prompt="Reply with READY.",
        temperature=0.0,
    )

    assert post.call_args.kwargs["json"]["temperature"] == 0.0


def test_chat_completion_retries_empty_response_with_larger_token_limit(monkeypatch):
    monkeypatch.setenv("AZURE_INFERENCE_ENDPOINT", "https://example.services.ai.azure.com")
    monkeypatch.setattr(inference, "get_azure_inference_token", lambda _: "token")
    empty_response = Mock()
    empty_response.json.return_value = {
        "choices": [{"finish_reason": "length", "message": {"content": ""}}]
    }
    valid_response = Mock()
    valid_response.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": "READY"}}]
    }
    post = Mock(side_effect=[empty_response, valid_response])
    monkeypatch.setattr(inference.requests, "post", post)

    result = inference.chat_completion(
        credential=Mock(),
        model="gpt-5-mini",
        prompt="Reply with READY.",
        max_tokens=800,
    )

    assert result == "READY"
    assert post.call_count == 2
    assert post.call_args_list[1].kwargs["json"]["max_completion_tokens"] == 1600


def test_chat_completion_raises_when_retry_is_still_empty(monkeypatch):
    monkeypatch.setenv("AZURE_INFERENCE_ENDPOINT", "https://example.services.ai.azure.com")
    monkeypatch.setattr(inference, "get_azure_inference_token", lambda _: "token")
    response = Mock()
    response.json.return_value = {
        "choices": [{"finish_reason": "length", "message": {"content": ""}}]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(inference.requests, "post", post)

    with pytest.raises(RuntimeError, match="no response text after one retry"):
        inference.chat_completion(
            credential=Mock(),
            model="gpt-5-mini",
            prompt="Reply with READY.",
            max_tokens=800,
        )
