from tinker_cookbook.recipes.kokkos_rl.train_kokkos import CLIConfig, _to_harbor_config


def test_config_translation_preserves_training_fields() -> None:
    cli = CLIConfig(
        model_name="openai/gpt-oss-120b:peft:131072",
        group_size=8,
        groups_per_batch=16,
        learning_rate=2e-5,
        max_turns=30,
        max_steps=7,
        max_steps_off_policy=2,
        thinking_effort=0.9,
    )
    harbor = _to_harbor_config(cli)

    assert harbor.model_name == cli.model_name
    assert harbor.group_size == 8
    assert harbor.groups_per_batch == 16
    assert harbor.learning_rate == 2e-5
    assert harbor.max_turns == 30
    assert harbor.max_steps == 7
    assert harbor.max_steps_off_policy == 2
    # thinking_effort must be forwarded so Inkling rollouts keep the pinned
    # effort from the Phase 0 plan.
    assert harbor.thinking_effort == 0.9


def test_defaults_match_phase0_eval_budget() -> None:
    cli = CLIConfig()
    assert cli.tasks_dir == "data/kokkos/phase0/harbor"
    assert cli.max_trajectory_tokens == 112 * 1024
    assert cli.max_tokens == 16384
