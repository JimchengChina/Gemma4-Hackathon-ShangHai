from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Any, Callable

from .model import Gemma4Client, extract_json_object, parse_tool_calls
from .schemas import EvidenceFrame, LiabilityResult
from .tools import (
    append_audit_log,
    calibrated_ratio_head,
    extract_video_metadata,
    preprocess_multimodal_case,
    redact_text,
    retrieve_law_articles,
)


ToolFn = Callable[..., Any]


def build_tool_schemas(functions: list[ToolFn]) -> list[dict[str, Any]]:
    """Build JSON schemas for Gemma 4 native function calling."""
    try:
        from transformers.utils import get_json_schema
    except Exception as exc:
        raise RuntimeError("transformers.utils.get_json_schema is required for tool schemas.") from exc

    return [get_json_schema(fn) for fn in functions]


def call_safe_tool(name: str, arguments: dict[str, Any], registry: dict[str, ToolFn]) -> Any:
    """Execute only registered local tools."""
    if name not in registry:
        raise ValueError(f"Unauthorized tool call: {name}")
    return registry[name](**arguments)


class TrafficLiabilityPipeline:
    """Multimodal traffic liability pipeline using Gemma 4 as the reasoning hub."""

    def __init__(self, client: Gemma4Client | None = None) -> None:
        self.client = client or Gemma4Client()
        self.tool_registry: dict[str, ToolFn] = {
            "extract_video_metadata": extract_video_metadata,
            "preprocess_multimodal_case": preprocess_multimodal_case,
            "retrieve_law_articles": retrieve_law_articles,
            "calibrated_ratio_head": calibrated_ratio_head,
        }
        self.tool_schemas = build_tool_schemas(list(self.tool_registry.values()))

    def summarize_video_in_chunks(
        self,
        case_id: str,
        frame_records: list[EvidenceFrame],
        chunk_size: int = 4,
    ) -> list[dict[str, Any]]:
        """Summarize sampled video frames with multimodal Gemma 4 calls."""
        from PIL import Image

        summaries: list[dict[str, Any]] = []
        for offset in range(0, len(frame_records), chunk_size):
            chunk = frame_records[offset:offset + chunk_size]
            images = [Image.open(frame.image_path).convert("RGB") for frame in chunk]
            time_hints = ", ".join(f"{frame.timestamp_sec:.1f}s" for frame in chunk)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an accident evidence analyst. Only describe visible facts. "
                        "Return JSON with observed_facts, actors, actions and uncertainties."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Frame timestamps: {time_hints}. Summarize visible facts."},
                        *[{"type": "image"} for _ in images],
                    ],
                },
            ]

            try:
                raw = self.client.generate(messages=messages, images=images, enable_thinking=False)
            except Exception as exc:
                raw = json.dumps(
                    {
                        "observed_facts": [],
                        "actors": [],
                        "actions": [],
                        "uncertainties": [f"chunk_summary_error: {exc}"],
                    },
                    ensure_ascii=False,
                )

            summaries.append(
                {
                    "chunk_index": offset // chunk_size,
                    "start_ts": chunk[0].timestamp_sec,
                    "end_ts": chunk[-1].timestamp_sec,
                    "summary_raw": raw,
                }
            )

        append_audit_log(
            {
                "case_id": case_id,
                "stage": "chunk_summarization",
                "chunks": len(summaries),
                "model_id": self.client.model_id,
            }
        )
        return summaries

    def run(
        self,
        video_path: str,
        text_report: str,
        image_paths: list[str],
        party_names: list[str],
        max_tool_steps: int = 4,
    ) -> LiabilityResult:
        """Run the full evidence-to-liability assistant pipeline."""
        case_id = uuid.uuid4().hex
        prep = preprocess_multimodal_case(video_path, text_report, image_paths)
        frames = [EvidenceFrame(**record) for record in prep["sampled_frames"]]
        chunk_summaries = self.summarize_video_in_chunks(case_id, frames)
        law_hits = retrieve_law_articles(query=redact_text(text_report), top_k=5)

        tool_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a traffic accident liability assistant. Use tools for verifiable facts. "
                    "Do not invent law articles or unseen facts. If evidence is insufficient, mark "
                    "uncertainties and require human review."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"case_id: {case_id}\n"
                    f"parties: {json.dumps(party_names, ensure_ascii=False)}\n"
                    f"redacted_report: {prep['redacted_text_report']}\n"
                    f"video_metadata: {json.dumps(prep['video_metadata'], ensure_ascii=False)}\n"
                    f"chunk_timeline: {json.dumps(chunk_summaries, ensure_ascii=False)}\n"
                    f"law_hits: {json.dumps(law_hits, ensure_ascii=False)}\n"
                    "Decide whether local tools are needed before the final liability JSON."
                ),
            },
        ]

        for step in range(max_tool_steps):
            assistant_output = self.client.generate(
                messages=tool_messages,
                tools=self.tool_schemas,
                enable_thinking=False,
                max_new_tokens=512,
            )
            tool_messages.append({"role": "assistant", "content": assistant_output})
            calls = parse_tool_calls(assistant_output)
            if not calls:
                break

            for call in calls:
                try:
                    tool_result = call_safe_tool(call.name, call.arguments, self.tool_registry)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "name": call.name,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    append_audit_log(
                        {
                            "case_id": case_id,
                            "stage": "tool_call",
                            "step": step,
                            "tool_name": call.name,
                            "tool_arguments": call.arguments,
                            "tool_result_preview": str(tool_result)[:500],
                            "model_id": self.client.model_id,
                        }
                    )
                except Exception as exc:
                    tool_messages.append(
                        {
                            "role": "tool",
                            "name": call.name,
                            "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                        }
                    )
                    append_audit_log(
                        {
                            "case_id": case_id,
                            "stage": "tool_error",
                            "step": step,
                            "tool_name": call.name,
                            "tool_arguments": call.arguments,
                            "error": str(exc),
                            "model_id": self.client.model_id,
                        }
                    )

        final_messages = tool_messages + [
            {
                "role": "user",
                "content": (
                    "Return final JSON only. Required fields: accident_detected, parties, "
                    "liability_bucket, recommended_ratio_bucket, evidence_chain, "
                    "supporting_articles, uncertainties, human_review_required, model_opinion. "
                    "Only cite visible facts or tool outputs. Use Chinese for the report fields."
                ),
            }
        ]

        final_output = self.client.generate(
            messages=final_messages,
            enable_thinking=False,
            max_new_tokens=1024,
        )

        try:
            final_json = extract_json_object(final_output)
        except Exception:
            final_json = {
                "accident_detected": True,
                "parties": party_names,
                "liability_bucket": "同责",
                "recommended_ratio_bucket": "同责",
                "evidence_chain": [],
                "supporting_articles": law_hits,
                "uncertainties": ["模型输出不是标准 JSON，已进入保守兜底策略。"],
                "human_review_required": True,
                "model_opinion": "模型输出解析失败，建议人工复核。",
            }

        bucket = final_json.get("recommended_ratio_bucket") or final_json.get("liability_bucket") or "同责"
        ratio = calibrated_ratio_head(bucket, party_names)

        result = LiabilityResult(
            case_id=case_id,
            accident_detected=bool(final_json.get("accident_detected", True)),
            parties=final_json.get("parties", party_names),
            liability_ratio=ratio,
            liability_bucket=final_json.get("liability_bucket", bucket),
            evidence_chain=final_json.get("evidence_chain", []),
            supporting_articles=final_json.get("supporting_articles", law_hits),
            uncertainties=final_json.get("uncertainties", []),
            human_review_required=bool(final_json.get("human_review_required", True)),
            model_opinion=final_json.get("model_opinion", "未生成简洁结论。"),
        )

        append_audit_log(
            {
                "case_id": case_id,
                "stage": "pipeline_done",
                "result": asdict(result),
                "model_id": self.client.model_id,
            }
        )
        return result
