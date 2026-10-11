import json

import httpx
import pytest
from openai import AsyncOpenAI

from tinker_cookbook.recipes.kokkos_rl import chat_inference


@pytest.mark.asyncio
async def test_generic_provider_has_no_tinker_parameters(tmp_path, monkeypatch):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "portable",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "{}"},
                    }
                ],
            },
        )

    client = AsyncOpenAI(
        api_key="test-secret",
        base_url="https://provider.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(chat_inference, "create_chat_client", lambda *args: client)
    completer = chat_inference.ChatAnnotationCompleter(
        model="portable", max_tokens=10, thinking_effort=0.7, reports_dir=tmp_path, extra_body={}
    )
    await completer([{"role": "user", "content": "annotate"}])
    await completer.close()
    assert "reasoning_effort" not in requests[0]
    assert "separate_reasoning" not in requests[0]
    assert "test-secret" not in next(tmp_path.glob("*.json")).read_text()


@pytest.mark.parametrize(
    "url", ["https://key@provider.invalid/v1", "https://provider.invalid/v1?api_key=secret"]
)
def test_endpoint_cannot_leak_embedded_credentials(url):
    with pytest.raises(ValueError, match="embedded credentials"):
        chat_inference.create_chat_client(url)
