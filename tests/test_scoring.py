import math

import pytest

from llm_classifier.scoring import (
    choice_confidence,
    fit_decision_temperature,
    render,
    sigmoid,
    softmax,
    score_confidence,
)


def test_render_string_passthrough():
    assert render("hello") == "hello"


def test_render_dict_json_sorted():
    assert render({"b": 2, "a": 1}) == '{"a": 1, "b": 2}'


def test_sigmoid_zero_is_half():
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_approaches_one():
    assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative_approaches_zero():
    assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)


def test_softmax_empty_raises():
    with pytest.raises(ValueError):
        softmax([])


def test_softmax_invalid_temperature_raises():
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0)
    with pytest.raises(ValueError):
        softmax([1.0], temperature=float("nan"))


def test_softmax_uniform_equal_scores():
    assert softmax([0.0, 0.0, 0.0]) == pytest.approx([1 / 3] * 3)


def test_softmax_higher_score_higher_probability():
    probabilities = softmax([0.0, 2.0], temperature=1.0)
    assert probabilities[1] > probabilities[0]
    assert sum(probabilities) == pytest.approx(1.0)


def test_softmax_temperature_smooths_distribution():
    sharp = softmax([0.0, 2.0], temperature=1.0)
    smooth = softmax([0.0, 2.0], temperature=10.0)
    assert abs(sharp[1] - sharp[0]) > abs(smooth[1] - smooth[0])


def test_choice_confidence_single():
    assert choice_confidence([1.0]) == 1.0


def test_choice_confidence_range():
    assert choice_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert choice_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)


def test_score_confidence_single():
    assert score_confidence([1.0]) == 1.0


def test_score_confidence_uniform_is_zero():
    assert score_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0)


def test_score_confidence_sharp_high():
    assert score_confidence([0.0, 0.0, 1.0]) == pytest.approx(1.0)


def test_fit_decision_temperature_rejects_empty():
    with pytest.raises(ValueError):
        fit_decision_temperature([])


def test_fit_decision_temperature_prefers_sharp_distribution():
    # Target selalu index 1 dengan gap besar -> temperature kecil optimal.
    samples = [([0.0, 10.0], 1) for _ in range(5)]
    temperature = fit_decision_temperature(samples)
    # grid dimulai dari exp(-3) ~= 0.0498; distribusi tajam -> suhu bawah
    assert math.exp(-3.0) <= temperature <= 0.2


def test_fit_decision_temperature_result_in_grid_range():
    samples = [([0.0, 0.5], 1), ([0.5, 0.0], 0)]
    temperature = fit_decision_temperature(samples)
    assert math.exp(-3.0) <= temperature <= math.exp(3.0)
