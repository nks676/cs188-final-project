"""Unit tests for taskb/llm.py."""

import sys
import types

import pytest

import taskb.llm as llm
from taskb.llm import MODEL_NAME, CodeGenerationError, _extract_code, generate_code


class TestExtractCode:
    def test_extract_code_from_fenced_python_block(self):
        text = "```python\npick_and_place(0, [0.1, 0.2, 0.82])\n```"
        assert _extract_code(text) == "pick_and_place(0, [0.1, 0.2, 0.82])"

    def test_extract_code_from_unfenced_text(self):
        text = "pick_and_place(0, [0.1, 0.2, 0.82])"
        assert _extract_code(text) == text


class TestGenerateCode:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setattr(llm, "load_dotenv", lambda: None)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(CodeGenerationError, match="GEMINI_API_KEY"):
            generate_code("Put the red block in the corner.")

    def test_generate_code_returns_checked_response(self, monkeypatch):
        handle = self._install_fake_genai(monkeypatch, ["```python\nx = 1\n```"])
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        code = generate_code("Do something simple.")

        assert code == "x = 1"
        assert handle.client.calls == [
            {
                "model": MODEL_NAME,
                "contents_contains": "Do something simple.",
            }
        ]

    def test_generate_code_retries_after_syntax_error(self, monkeypatch):
        handle = self._install_fake_genai(
            monkeypatch,
            ["```python\ndef broken(:\n```", "```python\nx = 1\n```"],
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        code = generate_code("Retry please.", max_retries=1)

        assert code == "x = 1"
        assert len(handle.client.calls) == 2
        assert "Previous attempt had a syntax error" in handle.client.prompts[1]

    def test_generate_code_raises_on_safety_error_without_retry(self, monkeypatch):
        handle = self._install_fake_genai(monkeypatch, ["```python\nopen('bad')\n```"])
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with pytest.raises(CodeGenerationError, match="Blocked function call"):
            generate_code("Do a bad thing.", max_retries=2)

        assert len(handle.client.calls) == 1

    @staticmethod
    def _install_fake_genai(monkeypatch, responses):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

        class FakeModels:
            def __init__(self, client):
                self._client = client

            def generate_content(self, *, model, contents):
                self._client.calls.append(
                    {
                        "model": model,
                        "contents_contains": contents.splitlines()[-1].replace("# INSTRUCTION: ", ""),
                    }
                )
                self._client.prompts.append(contents)
                return FakeResponse(self._client.responses.pop(0))

        class FakeClient:
            def __init__(self, api_key):
                self.api_key = api_key
                self.responses = list(responses)
                self.calls = []
                self.prompts = []
                self.models = FakeModels(self)

        holder = types.SimpleNamespace(client=None)

        def client_factory(*, api_key):
            holder.client = FakeClient(api_key)
            return holder.client

        fake_google = types.ModuleType("google")
        fake_google.genai = types.SimpleNamespace(Client=client_factory)
        monkeypatch.setitem(sys.modules, "google", fake_google)
        return holder
