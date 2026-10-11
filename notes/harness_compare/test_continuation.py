import unittest

from continuation import continuation_reason, sum_usage


QUALITY = {
    "kind": "thinking_only",
    "assistant_stop_reason": "end_turn",
    "cli_result_subtype": "success",
    "cli_result_blank": True,
}


class ContinuationTest(unittest.TestCase):
    def decision(self, **overrides):
        args = dict(
            policy="thinking_only_resume", completed_continuations=0,
            max_continuations=3, model_turns=6, max_model_turns=40,
            tool_calls=5, max_tool_calls=80, remaining_seconds=120,
        )
        args.update(overrides)
        return continuation_reason(QUALITY, **args)

    def test_resumes_only_incomplete_thinking_end_turn(self):
        self.assertEqual(self.decision(), "resume")
        self.assertEqual(
            continuation_reason({**QUALITY, "assistant_stop_reason": None},
                                policy="thinking_only_resume", completed_continuations=0,
                                max_continuations=3, model_turns=6, max_model_turns=40,
                                tool_calls=5, max_tool_calls=80, remaining_seconds=120),
            "resume",
        )
        self.assertEqual(self.decision(policy="none"), "disabled")
        self.assertEqual(
            continuation_reason({**QUALITY, "kind": "final_text"},
                                policy="thinking_only_resume", completed_continuations=0,
                                max_continuations=3, model_turns=6, max_model_turns=40,
                                tool_calls=5, max_tool_calls=80, remaining_seconds=120),
            "other_completion",
        )
        self.assertEqual(
            continuation_reason({**QUALITY, "cli_result_blank": False},
                                policy="thinking_only_resume", completed_continuations=0,
                                max_continuations=3, model_turns=6, max_model_turns=40,
                                tool_calls=5, max_tool_calls=80, remaining_seconds=120),
            "cli_not_blank_success",
        )

    def test_preserves_episode_budgets(self):
        self.assertEqual(self.decision(completed_continuations=3), "continuation_limit")
        self.assertEqual(self.decision(model_turns=40), "model_turn_budget")
        self.assertEqual(self.decision(tool_calls=80), "tool_call_budget")
        self.assertEqual(self.decision(remaining_seconds=0), "wall_time_budget")

    def test_usage_is_aggregated_by_invocation(self):
        self.assertEqual(sum_usage([
            {"usage": {"input_tokens": 10, "output_tokens": 4}},
            {"usage": {"input_tokens": 20, "output_tokens": 6,
                       "cache_read_input_tokens": 5}},
        ]), {"input_tokens": 30, "output_tokens": 10,
             "cache_read_input_tokens": 5})


if __name__ == "__main__":
    unittest.main()
