from tinker_cookbook.rl.train import _should_evaluate


def test_periodic_and_final_evaluation_cadence() -> None:
    assert _should_evaluate(0, 10)
    assert _should_evaluate(10, 10)
    assert not _should_evaluate(19, 10)
    assert _should_evaluate(20, 10, is_final=True)
    assert _should_evaluate(17, 10, is_final=True)


def test_final_evaluation_is_deduplicated_and_can_be_disabled() -> None:
    assert not _should_evaluate(20, 10, is_final=True, last_evaluated_step=20)
    assert not _should_evaluate(20, 0, is_final=True)
