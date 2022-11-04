import math


def info(p: float) -> float:
    """Return information of the input probability p"""
    if not 0 <= p <= 1:
        raise ValueError("Probability should be within [0,1]")
    return math.log2(1 / p)


def entropy(pmf: list[float]) -> float:
    """Return expected value of information of the input PMF of a random variable"""
    if abs(sum(pmf) - 1) > 1e-5:
        raise ValueError("sum of PMF should be 1")
    result = 0
    for p in pmf:
        if p != 0:
            result += info(p) * p
    return result
