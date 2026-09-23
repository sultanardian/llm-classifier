import json
import math


def render(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sigmoid(value):
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax(values, temperature=1.0):
    if not values:
        raise ValueError("values tidak boleh kosong")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature harus positif dan finite")

    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def choice_confidence(probabilities):
    if len(probabilities) == 1:
        return 1.0
    uniform = 1.0 / len(probabilities)
    return (max(probabilities) - uniform) / (1.0 - uniform)


def score_confidence(probabilities):
    count = len(probabilities)
    if count == 1:
        return 1.0
    mode = max(range(count), key=probabilities.__getitem__)
    distance = sum(
        probability * abs(index - mode)
        for index, probability in enumerate(probabilities)
    )
    center = (count - 1) / 2
    uniform_deviation = sum(
        abs(index - center) for index in range(count)
    ) / count
    return max(0.0, 1.0 - distance / uniform_deviation)


def _cross_entropy(samples, temperature):
    total = 0.0
    for scores, target_index in samples:
        probabilities = softmax(scores, temperature)
        total -= math.log(max(probabilities[target_index], 1e-15))
    return total / len(samples)


def fit_decision_temperature(samples):
    if not samples:
        raise ValueError("samples tidak boleh kosong")

    # Grid log-spaced sederhana, cukup untuk baseline calibration.
    candidates = [
        math.exp(-3.0 + index * (6.0 / 120.0))
        for index in range(121)
    ]
    return min(
        candidates,
        key=lambda temperature: _cross_entropy(samples, temperature),
    )
