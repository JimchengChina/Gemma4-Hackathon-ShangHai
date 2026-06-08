from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gemma4_liability_assistant.model import extract_json_object, parse_tool_calls  # noqa: E402
from gemma4_liability_assistant.tools import calibrated_ratio_head, redact_text  # noqa: E402


class ToolCallTests(unittest.TestCase):
    def test_parse_native_tool_call(self) -> None:
        text = '<|tool_call>call:retrieve_law_articles{query:<|"|>rear-end collision<|"|>,top_k:3}<tool_call|>'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "retrieve_law_articles")
        self.assertEqual(calls[0].arguments["query"], "rear-end collision")
        self.assertEqual(calls[0].arguments["top_k"], 3)

    def test_parse_list_argument(self) -> None:
        text = '<|tool_call>call:calibrated_ratio_head{candidate_bucket:<|"|>主责<|"|>,parties:["甲车","乙车"]}<tool_call|>'
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0].arguments["candidate_bucket"], "主责")
        self.assertEqual(calls[0].arguments["parties"], ["甲车", "乙车"])

    def test_extract_json_from_fenced_response(self) -> None:
        payload = extract_json_object('```json\n{"liability_bucket":"主责"}\n```')
        self.assertEqual(payload["liability_bucket"], "主责")

    def test_ratio_template(self) -> None:
        ratio = calibrated_ratio_head("主责", ["甲车", "乙车"])
        self.assertEqual(ratio, {"甲车": 0.7, "乙车": 0.3})

    def test_redact_text(self) -> None:
        redacted = redact_text("联系电话 13812345678，车牌 粤B12345")
        self.assertIn("[PHONE]", redacted)
        self.assertIn("[PLATE]", redacted)


if __name__ == "__main__":
    unittest.main()
