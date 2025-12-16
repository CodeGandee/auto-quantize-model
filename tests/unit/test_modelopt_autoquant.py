from __future__ import annotations

from auto_quantize_model.modelopt_autoquant import compute_num_score_steps


def test_compute_num_score_steps_clamps_to_num_batches() -> None:
    assert compute_num_score_steps(score_size=128, batch_size=8, num_batches=4) == 4


def test_compute_num_score_steps_minimum_is_one() -> None:
    assert compute_num_score_steps(score_size=1, batch_size=8, num_batches=10) == 1
    assert compute_num_score_steps(score_size=0, batch_size=8, num_batches=10) == 1
    assert compute_num_score_steps(score_size=128, batch_size=8, num_batches=0) == 1

