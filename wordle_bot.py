import math
import json
import wordle_json


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
        if p != 0:  # avoid ZeroDivisionError
            result += info(p) * p
    return result


def eliminate(guess: str, pattern: str, n: int = 0) -> None:
    """Eliminate words that does not comply with the guess and the pattern. n is the number of times of attempt.
    store the result in a new txt file, and create the corresponding JSON file."""
    with open(f"input_mass_function_{n - 1}.json", "r") as in_f, open(f"word_list_{n}.txt", "w") as out_f:
        word_list = json.load(in_f)[guess][pattern]
        for word in word_list:
            print(word, file=out_f)
    wordle_json.create_jsons(n)


def greedy_find(disp: int = 10, n: int = 0) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in a single step.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and entropy in descending order."""
    with open(f"pmfs_{n}.json", "r") as f:
        pmfs = json.load(f)
        entropy_list = []
        for guess, pmf in pmfs.items():
            entropy_list.append((guess, entropy(pmf.values())))
    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def two_steps_find(disp: int = 10, n: int = 0) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in two steps.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and sum of entropy of two steps in descending order."""
    with open(f"pmfs_{n}.json", "r") as f1:
        pmfs = json.load(f1)
        entropy_list = []
        for guess, pmf in pmfs.items():
            sec_ord_entropy = 0  # define the entropy at the second step
            for pattern, p in pmf.items():  # find maximum entropy at the second step
                eliminate(guess, pattern, n + 1)
                wordle_json.create_jsons(n + 1)
                sec_ord_entropy += pmfs[guess][pattern] * greedy_find(1, n + 1)[0][1]  # weighted average of entropy
            entropy_list.append((guess, entropy(pmf.values()), sec_ord_entropy,
                                 entropy(pmf.values()) + sec_ord_entropy))
        entropy_list.sort(key=lambda x: x[-1], reverse=True)
        return entropy_list[:disp] if disp != -1 else entropy_list


if __name__ == "__main__":
    print(greedy_find())
