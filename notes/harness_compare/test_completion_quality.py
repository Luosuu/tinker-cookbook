"""Regression tests for non-actionable successful Claude CLI exits."""

import json
import unittest

from completion_quality import classify_stream


def event(kind: str, **fields: object) -> str:
    return json.dumps({"type": kind, **fields})


class CompletionQualityTests(unittest.TestCase):
    def test_successful_cli_exit_can_still_be_thinking_only(self) -> None:
        lines = [
            event("assistant", message={"content": [{"type": "tool_use", "name": "mcp__hpc__bash"}]}),
            event("assistant", message={"content": [{"type": "thinking", "thinking": "Now edit the file."}]}),
            event("result", subtype="success", result=" "),
        ]
        self.assertEqual(classify_stream(lines), {
            "kind": "thinking_only",
            "content_types": ["thinking"],
            "assistant_stop_reason": None,
            "cli_result_subtype": "success",
            "cli_result_blank": True,
        })

    def test_tool_syntax_in_text_is_not_a_tool_call(self) -> None:
        lines = [
            event("assistant", message={"content": [{
                "type": "text", "text": 'Search.mcp__hpc__bash","args":{"command":"ls"}}',
            }]}),
            event("result", subtype="success", result=" "),
        ]
        self.assertEqual(classify_stream(lines)["kind"], "tool_call_as_text")

    def test_normal_final_text(self) -> None:
        lines = [
            event("assistant", message={"content": [{"type": "text", "text": "Done."}], "stop_reason": "end_turn"}),
            event("result", subtype="success", result="Done."),
        ]
        self.assertEqual(classify_stream(lines)["kind"], "final_text")


if __name__ == "__main__":
    unittest.main()
