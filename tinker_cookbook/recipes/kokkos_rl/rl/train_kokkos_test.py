import functools
from pathlib import Path

import pytest

from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory
from tinker_cookbook.recipes.kokkos_rl.rl.train_kokkos import (
    DEFAULT_CONTREE_CACHE_PATH,
    CLIConfig,
    _build_sandbox_factory,
    _to_harbor_config,
)


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


def test_default_sandbox_backend_is_modal_and_blocks_network() -> None:
    factory = _build_sandbox_factory(CLIConfig())

    assert isinstance(factory, functools.partial)
    assert factory.func is default_sandbox_factory
    assert factory.keywords == {"allow_network": False}


def test_heldout_and_gradient_hygiene_fields_are_forwarded() -> None:
    cli = CLIConfig(
        eval_size=20,
        eval_group_size=4,
        heldout_max_concurrent_groups=2,
        split_seed=7,
        remove_constant_reward_groups=True,
        raise_on_grading_error=True,
    )
    harbor = _to_harbor_config(cli)

    assert harbor.eval_size == 20
    assert harbor.eval_group_size == 4
    assert harbor.heldout_max_concurrent_groups == 2
    assert harbor.split_seed == 7
    assert harbor.remove_constant_reward_groups is True
    assert harbor.raise_on_grading_error is True


def test_heldout_defaults_preserve_legacy_behavior() -> None:
    harbor = _to_harbor_config(CLIConfig())

    assert harbor.eval_size == 0
    assert harbor.remove_constant_reward_groups is False
    assert harbor.raise_on_grading_error is False


def test_task_names_defaults_to_no_filtering() -> None:
    assert CLIConfig().task_names is None


def test_unknown_sandbox_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown sandbox_backend"):
        _build_sandbox_factory(CLIConfig(sandbox_backend="nebius"))


def test_contree_backend_uses_shared_image_cache(tmp_path) -> None:
    """The ConTree cache is keyed by Dockerfile digest, so it is shared by default."""
    from tinker_cookbook.sandbox.contree_sandbox import ContreeDockerfileSandboxFactory

    explicit = tmp_path / "images.json"
    factory = _build_sandbox_factory(
        CLIConfig(
            sandbox_backend="contree",
            contree_cache_path=str(explicit),
            sandbox_build_parallelism=2,
            sandbox_timeout=1800,
        )
    )
    assert isinstance(factory, ContreeDockerfileSandboxFactory)
    assert factory._cache_path == explicit
    assert factory._timeout == 1800
    assert factory._runtime_build_parallelism == 2
    assert factory._allow_network is False

    default_factory = _build_sandbox_factory(CLIConfig(sandbox_backend="contree"))
    assert default_factory._cache_path == Path(DEFAULT_CONTREE_CACHE_PATH)


def test_network_can_be_explicitly_enabled_for_both_backends(tmp_path) -> None:
    modal_factory = _build_sandbox_factory(CLIConfig(allow_network=True))
    assert isinstance(modal_factory, functools.partial)
    assert modal_factory.keywords == {"allow_network": True}

    contree_factory = _build_sandbox_factory(
        CLIConfig(
            sandbox_backend="contree",
            contree_cache_path=str(tmp_path / "images.json"),
            allow_network=True,
        )
    )
    assert contree_factory._allow_network is True
