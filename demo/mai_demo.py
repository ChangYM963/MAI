"""A synthetic MAI illustration; this does not reproduce model experiments."""

from math import isclose, isfinite


def mai(probabilities, majority):
    """Alignment on five ordered options, with a fixed majority reference."""
    if len(probabilities) != 5 or majority not in range(5):
        raise ValueError("Use five probabilities and a majority index from 0 to 4.")
    if any(not isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to one.")
    distance = max(majority, 4 - majority)
    return sum(4 * (1 - abs(k - majority) / distance) * p
               for k, p in enumerate(probabilities))


def main():
    # Hand-written examples, not responses from a model or research dataset.
    cases = [
        ("Independent",        [0.55, 0.20, 0.15, 0.05, 0.05]),
        ("Subtle drift",       [0.50, 0.20, 0.15, 0.10, 0.05]),
        ("Switch to majority", [0.10, 0.10, 0.20, 0.55, 0.05]),
    ]
    majority = 3  # Option D; held fixed in all three cases.
    initial = mai(cases[0][1], majority)
    print("Case                Modal choice    MAI    Delta MAI")
    for name, probabilities in cases:
        choice = "ABCDE"[max(range(5), key=probabilities.__getitem__)]
        score = mai(probabilities, majority)
        print(f"{name:<20}{choice:<14}{score:5.2f}{score - initial:+12.2f}")


if __name__ == "__main__":
    main()
