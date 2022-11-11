import json
import math
from time import time, gmtime, strftime

import numpy as np

import wordle_json as wj
from wordle import plot_pmf

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


def eliminate(guess: str, pattern: str, file: int | dict = 0, write: bool = True) -> None | dict:
    """Eliminate words that does not comply with the guess and the pattern. n is the number of times of attempt.
    store the result in a new txt file, and create the corresponding JSON file."""
    if isinstance(file, int):
        with open(f"input_mass_function_{file - 1}.json", "r") as in_f:
            words = json.load(in_f)[guess][pattern]
    else:
        write = False
        words = file[guess][pattern]
    print(f"data size: {len(words)}")
    # print(f"Number of remaining choices: {len(words)}")
    if write:
        with open(f"word_list_{file}.txt", "w") as out_f:
            for word in words:
                print(word, file=out_f)
        wj.create_data(file)
    else:
        return wj.create_data(np.array(words))


def one_step_greedy(n: int = 0, disp: int = 10, data=None) -> list[tuple]:
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


def two_step_greedy(n: int = 0, disp: int = 10, data=None) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in two steps.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and sum of entropy of two steps in descending order."""
    start_time = time()
    if data is None:
        with open(f"pmfs_{n}.json", "r") as f1, open(f"input_mass_function_{n}.json", "r") as f2:
            pmfs = json.load(f1)
            mass_func = json.load(f2)

    entropy_list = []
    total = 100  # maximum number of top words in one_step greedy to speed up the program.
    count = 0
    top_words = dict(one_step_greedy(n + 1, total, pmfs)).keys()
    for guess, pmf in pmfs.items():
        if guess not in top_words:
            continue
        sec_ord_entropy = 0  # define the entropy at the second step
        for pattern, p in pmf.items():  # find maximum entropy at the second step
            sub_pmfs = eliminate(guess, pattern, file=mass_func, write=False)
            sec_ord_entropy += pmfs[guess][pattern] * one_step_greedy(n + 1, 1, sub_pmfs)[0][1]
            # weighted average of entropy
        entropy_list.append((guess, entropy(pmf.values()) + sec_ord_entropy))
        count += 1
        print(f"Progress: {count}/{total} ({count / total * 100}%)")
        print(f"Time elapsed: {strftime('%H:%M:%S', gmtime(time() - start_time))}")
        print(f"Estimated time: {strftime(f'%H:%M:%S', gmtime((time() - start_time) / (count / total)))}")
    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def create_greedy() -> None:
    """Store the initial result of one_step_greedy and two_step_greedy in the corresponding JSON file
    to speed up the program."""
    # wj.create_data()
    one_step = one_step_greedy(disp=-1)
    with open("one_step_entropy.json", "w") as f:
        json.dump(dict(one_step), f, indent=4)
        print("one step done.")
    two_step = two_step_greedy(disp=-1)
    with open("two_step_entropy.json", "w") as f:
        json.dump(dict(two_step), f, indent=4)
        print("two step done.")


def bot() -> None:
    print(one_step_greedy())
    for i in range(1, 7):
        guess = input("Please input your guess:")
        pattern = input("please input the returned pattern:")
        plot_pmf(guess, pattern, i-1)
        if pattern == '22222':
            for i in range(1, i):
                wj.del_data(i)
            break
        eliminate(guess, pattern, i)
        print(one_step_greedy(i))

    print("fin.")


def simulator(start_word="tares") -> None:
    pass


if __name__ == "__main__":
    bot()
