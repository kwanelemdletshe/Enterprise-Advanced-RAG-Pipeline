import os
from typing import List

import requests
from azure.core.credentials import TokenCredential
from dotenv import load_dotenv


load_dotenv()


def get_azure_inference_token(
    credential: TokenCredential,
) -> str:
    return credential.get_token(
        "https://cognitiveservices.azure.com/.default"
    ).token


def embed_texts(
    credential: TokenCredential,
    model: str,
    texts: List[str],
    dimensions: int = 1536,
) -> List[List[float]]:
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_INFERENCE_ENDPOINT must be configured.")
    url = f"{endpoint.rstrip('/')}/models/embeddings?api-version=2024-05-01-preview"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {get_azure_inference_token(credential)}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": texts,
            "dimensions": dimensions,
        },
        timeout=120,
    )
    response.raise_for_status()
    return [item["embedding"] for item in response.json()["data"]]


def chat_completion(
    credential: TokenCredential,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> str:
    endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_INFERENCE_ENDPOINT must be configured.")
    url = f"{endpoint.rstrip('/')}/models/chat/completions?api-version=2024-05-01-preview"
    request_body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if not model.lower().startswith("gpt-5"):
        request_body["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {get_azure_inference_token(credential)}",
        "Content-Type": "application/json",
    }

    for attempt in range(2):
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            timeout=120,
        )
        response.raise_for_status()
        choices = response.json().get("choices")
        if not choices:
            raise RuntimeError("Azure chat completion returned no choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

        if attempt == 0:
            request_body["max_completion_tokens"] = max(max_tokens * 2, 1600)
            continue

        raise RuntimeError(
            "Azure chat completion returned no response text after one retry "
            f"(finish_reason={choice.get('finish_reason')!r})."
        )

    raise RuntimeError("Azure chat completion returned no response text.")
