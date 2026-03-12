"""Gemini Flash client with code extraction and retry logic."""
import os
import re

from dotenv import load_dotenv

from taskb.prompt import build_prompt
from taskb.sandbox import check_safety

MODEL_NAME = "gemini-2.5-flash"


class CodeGenerationError(Exception):
    pass


def _extract_code(text: str) -> str:
    """Strip markdown fences; fall back to full response text."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def generate_code(instruction: str, max_retries: int = 2) -> str:
    """
    Call Gemini Flash and return validated Python code for the instruction.

    Retries up to max_retries times on SyntaxError from AST parse.
    Raises CodeGenerationError if all retries fail.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise CodeGenerationError("GEMINI_API_KEY environment variable is not set.")

    from google import genai  # noqa: PLC0415

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(instruction)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            prompt = prompt + "\n# Previous attempt had a syntax error. Return only valid Python."

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        code = _extract_code(response.text)

        try:
            check_safety(code)
            return code
        except SyntaxError as exc:
            last_error = exc
        except Exception as exc:
            # Safety violations or other errors — don't retry
            raise CodeGenerationError(str(exc)) from exc

    raise CodeGenerationError(
        f"Failed to generate valid code after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
