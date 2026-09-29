"""Calibrate a lightweight risk head and the paper's projection intervention."""

import math


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Hidden vector dimensions differ")
    return sum(x * y for x, y in zip(a, b))


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


def fit_hidden_controller(examples, epochs=300, learning_rate=0.15):
    """Positive labels mean p(majority) rose; use only calibration observations.

    The reference head is standardized logistic regression. Direction estimation
    and the conditional projection update follow the paper's method.
    """
    if not examples:
        return {"available": False, "reason": "No valid calibration pairs"}
    positives = [row for row in examples if row["positive"]]
    if not positives:
        return {"available": False, "reason": "No positive calibration examples"}
    dimension = len(examples[0]["after"])
    direction = [sum(row["after"][i] - row["before"][i] for row in positives) / len(positives)
                 for i in range(dimension)]
    norm = math.sqrt(dot(direction, direction))
    if norm <= 1e-12:
        return {"available": False, "reason": "Positive examples have zero mean direction"}
    direction = [x / norm for x in direction]
    means = [sum(row["after"][i] for row in examples) / len(examples) for i in range(dimension)]
    scales = [max(1e-6, math.sqrt(sum((row["after"][i] - means[i]) ** 2 for row in examples) / len(examples)))
              for i in range(dimension)]
    features = [[(v - m) / s for v, m, s in zip(row["after"], means, scales)] for row in examples]
    weights, bias = [0.0] * dimension, 0.0
    for _ in range(epochs):
        gradient, bias_gradient = [0.0] * dimension, 0.0
        for x, row in zip(features, examples):
            residual = sigmoid(dot(weights, x) + bias) - float(row["positive"])
            bias_gradient += residual
            for i in range(dimension):
                gradient[i] += residual * x[i]
        weights = [w - learning_rate * (g / len(examples) + 0.001 * w) for w, g in zip(weights, gradient)]
        bias -= learning_rate * bias_gradient / len(examples)
    return {"available": True, "head": "standardized-logistic-reference", "direction": direction,
            "weights": weights, "bias": bias, "means": means, "scales": scales,
            "calibration_pairs": len(examples), "positive_pairs": len(positives)}


def hidden_update(hidden, controller, minority, alpha=0.6, threshold=0.4):
    if not math.isfinite(alpha) or alpha < 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid hidden intervention strength or threshold")
    if not controller.get("available"):
        return [0.0] * len(hidden), {"triggered": False, "reason": controller.get("reason", "Unavailable controller")}
    features = [(x - m) / s for x, m, s in zip(hidden, controller["means"], controller["scales"])]
    risk = sigmoid(dot(controller["weights"], features) + controller["bias"])
    projection = dot(hidden, controller["direction"])
    triggered = bool(minority and risk >= threshold and projection > 0 and alpha > 0)
    scale = -alpha * risk * max(0, projection) if triggered else 0.0
    return [scale * value for value in controller["direction"]], {
        "risk": risk, "projection": projection, "triggered": triggered,
        "alpha": alpha, "threshold": threshold,
    }
