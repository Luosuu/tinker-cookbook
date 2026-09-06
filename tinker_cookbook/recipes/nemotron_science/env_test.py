from unittest.mock import Mock

from tinker_cookbook.recipes.nemotron_science import env


class PlainRenderer:
    def build_generation_prompt(self, messages):
        return messages

    def create_conversation_prefix_with_tools(self, **kwargs):
        return []


class EffortRenderer(PlainRenderer):
    def build_generation_prompt(self, messages, effort=0.9):
        return effort


def test_effort_binding_leaves_other_renderers_callable():
    renderer = PlainRenderer()
    assert env._bind_effort(renderer, 0.3).build_generation_prompt(["hello"]) == ["hello"]


def test_effort_binding_conditions_every_generation():
    renderer = env._bind_effort(EffortRenderer(), 0.3)
    assert renderer.build_generation_prompt([]) == 0.3
    assert renderer.build_generation_prompt(["next"]) == 0.3


def test_judge_selects_its_own_renderer(monkeypatch):
    seen = []
    monkeypatch.setattr(env, "_JUDGE_CACHE", {})
    monkeypatch.setattr(env.tinker, "ServiceClient", Mock())
    monkeypatch.setattr(env, "LLMJudge", Mock())
    monkeypatch.setattr(env.tokenizer_utils, "get_tokenizer", lambda model: model)
    monkeypatch.setattr(
        env.model_info,
        "get_recommended_renderer_name",
        lambda model: "tml_v0" if model.startswith("thinkingmachines/") else "qwen3",
    )

    def get_renderer(name, tokenizer):
        seen.append((name, tokenizer))
        return EffortRenderer() if name == "tml_v0" else PlainRenderer()

    monkeypatch.setattr(env, "get_renderer", get_renderer)
    monkeypatch.setattr(env, "build_agent_tool_env", lambda **kwargs: kwargs)
    env.build_science_env(
        datum=env.ScienceDatum("question", "answer", None),
        model_name="thinkingmachines/Inkling-Small",
        renderer_name=None,
        judge_model="Qwen/Qwen3-8B",
        max_turns=8,
        max_tool_calls=8,
        max_trajectory_tokens=1000,
        format_coef=0.1,
        trace_weave=False,
        split="train",
    )
    assert seen == [("tml_v0", "thinkingmachines/Inkling-Small"), ("qwen3", "Qwen/Qwen3-8B")]
