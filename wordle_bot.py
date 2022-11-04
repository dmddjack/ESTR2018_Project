import math


def info(p: float) -> float:
    """Return information of the input probability p"""
    return math.log2(1 / p)


def entropy(pmf: list[float]) -> float:
    """Return expected value of information of the input PMF of a random variable"""
    result = 0
    for p in pmf:
        result += info(p) * p
    return result

