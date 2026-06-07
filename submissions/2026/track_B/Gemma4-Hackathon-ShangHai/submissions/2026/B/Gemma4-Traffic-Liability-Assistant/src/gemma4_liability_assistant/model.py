from __future__ import annotations

import json
import os
import re
import ast
from typing import Any

from .schemas import ToolCall


TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call>call:(?P<name>[A-Za-z_]\w*)\{(?P<args>.*?)\}<tool_call\|>",
    re.DOTALL,
)
QUOTE_MARKER = '<|"|>'


def _cast_tool_arg(value: str) -> Any:
    raw = value.strip().strip("'\"")
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if raw.startswith("[") or raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return raw

    try:
        return int(raw)
    except ValueError:
        pass

    try:
        return float(raw)
    except ValueError:
        return raw


def _split_top_level_args(raw_args: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_tool_quote = False
    i = 0

    while i < len(raw_args):
        if raw_args.startswith(QUOTE_MARKER, i):
            in_tool_quote = not in_tool_quote
            i += len(QUOTE_MARKER)
            continue

        char = raw_args[i]
        if not in_tool_quote:
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue

        current.append(char)
        i += 1

    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _parse_tool_args(raw_args: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for part in _split_top_level_args(raw_args):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_]\w*", key):
            args[key] = _cast_tool_arg(value)
    return args


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse Gemma 4 function-calling text into structured tool calls.

    Official processors may expose richer parsing helpers. This parser is kept as
    a small, auditable fallback for the native `<|tool_call>call:name{...}` form.
    """
    calls: list[ToolCall] = []

    for match in TOOL_CALL_PATTERN.finditer(text):
        name = match.group("name")
        raw_args = match.group("args")
        calls.append(ToolCall(name=name, arguments=_parse_tool_args(raw_args)))

    return calls


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start:end + 1])
    raise ValueError("No JSON object found in model response.")


class Gemma4Client:
    """Thin wrapper around a Hugging Face Gemma 4 multimodal model."""

    def __init__(
        self,
        model_id: str | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id or os.getenv("GEMMA_MODEL_ID", "google/gemma-4-12B-it")

        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForMultimodalLM as ModelClass

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = ModelClass.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=device_map,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = False,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
    ) -> str:
        """Run Gemma 4 with optional native tools and multimodal image input."""
        import torch

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )

        processor_args: dict[str, Any] = {
            "text": prompt,
            "return_tensors": "pt",
        }
        if images:
            processor_args["images"] = images

        inputs = self.processor(**processor_args)
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        return self.processor.decode(output[0][input_len:], skip_special_tokens=False)
