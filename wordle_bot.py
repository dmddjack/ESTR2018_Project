import math
import json


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


def greedy_find(n: int = 10) -> list:
    with open("pmfs.json", "r") as f:
        pmfs = json.load(f)
        entropy_list = []
        for guess, pmf in pmfs.items():
            entropy_list.append((guess, entropy(pmf.values())))
    entropy_list.sort(key=lambda x: x[1], reverse=True)
    return entropy_list[:n]


def recur_find() -> str:
    pass


if __name__ == "__main__":
    print(greedy_find())
