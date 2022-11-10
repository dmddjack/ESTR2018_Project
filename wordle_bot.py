import json
import math

import numpy as np

import wordle_json


def info(p: float | np.ndarray) -> float:
    """Return information of the input probability p"""
    if isinstance(p, np.ndarray):
        if ((0 > p) | (p > 1)).any():
            raise ValueError("Probability should be within [0,1]")
        return np.log2(1 / p)
    else:
        if not 0 <= p <= 1:
            raise ValueError("Probability should be within [0,1]")
        return math.log2(1 / p)


def entropy(pmf: list[float]) -> float:
    """Return expected value of information of the input PMF of a random variable"""
    pmf = np.fromiter(pmf, dtype=float)
    pmf = pmf[pmf != 0]  # filter zero values to avoid ZeroDivisionError
    if np.abs(np.sum(pmf) - 1) > 1e-5:
        raise ValueError("sum of PMF should be 1")
    result = np.dot(info(pmf), pmf)
    return result


def eliminate(guess: str, pattern: str, n: int = 0, write: bool = True) -> None | dict:
    """Eliminate words that does not comply with the guess and the pattern. n is the number of times of attempt.
    store the result in a new txt file, and create the corresponding JSON file."""

    with open(f"input_mass_function_{n - 1}.json", "r") as in_f:
        words = json.load(in_f)[guess][pattern]
        if write:
            with open(f"word_list_{n}.txt", "w") as out_f:
                for word in words:
                    print(word, file=out_f)
            wordle_json.create_data(n)
        else:
            return wordle_json.create_data(np.array(words))


def one_step_greedy(disp: int = 10, n: int = 0, data=None) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in a single step.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and entropy in descending order."""

    def find_entropy(pmfs: dict) -> list[tuple]:
        result = []
        for guess, pmf in pmfs.items():
            result.append((guess, entropy(pmf.values())))
        return result

    if data is None:
        with open(f"pmfs_{n}.json", "r") as f:
            entropy_list = find_entropy(json.load(f))
    else:
        entropy_list = find_entropy(data)

    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def two_step_greedy(disp: int = 10, n: int = 0) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in two steps.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and sum of entropy of two steps in descending order."""
    with open(f"pmfs_{n}.json", "r") as f1:
        pmfs = json.load(f1)
        entropy_list = []
        total, count = len(pmfs), 0

        for guess, pmf in pmfs.items():
            sec_ord_entropy = 0  # define the entropy at the second step
            for pattern, p in pmf.items():  # find maximum entropy at the second step
                sub_pmfs = eliminate(guess, pattern, n + 1, write=False)
                sec_ord_entropy += pmfs[guess][pattern] * one_step_greedy(1, n + 1, sub_pmfs)[0][1]
                # weighted average of entropy
            entropy_list.append((guess, entropy(pmf.values()), sec_ord_entropy,
                                 entropy(pmf.values()) + sec_ord_entropy))
            count += 1
            print(f"Progress: {count}/{total} ({count / total * 100}%)")
        entropy_list.sort(key=lambda x: x[-1], reverse=True)
        return entropy_list[:disp] if disp != -1 else entropy_list


if __name__ == "__main__":
    print(two_step_greedy())
