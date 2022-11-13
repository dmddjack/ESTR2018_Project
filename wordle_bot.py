import json
import math
from time import time, gmtime, strftime

import numpy as np

from wordle import check_word, plot_pmf
from wordle_json import create_json, create_data, del_data


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


def timer(start_time, progress, total) -> int:
    """Use time.time() to get the start time. Print the current progress and the estimated time.
    Return value updates the progress variable."""
    print(f"Progress: {progress}/{total} ({progress / total * 100}%)")
    print(f"Time elapsed: {strftime('%H:%M:%S', gmtime(time() - start_time))}")
    print(f"Estimated time: {strftime(f'%H:%M:%S', gmtime((time() - start_time) / (progress / total)))}")
    return progress + 1


def eliminate(guess: str, pattern: str, file: int | dict = 0, write: bool = True, step=1) -> None | dict:
    """Eliminate words that does not comply with the guess and the pattern. n is the number of times of attempt.
    store the result in a new txt file, and create the corresponding JSON file."""
    if isinstance(file, int):
        with open(f"input_mass_function_{file - 1}.json", "r") as in_f:
            words = json.load(in_f)[guess][pattern]
    else:
        write = False
        words = file[guess][pattern]
    # print(f"data size: {len(words)}")
    # print(f"Number of remaining choices: {len(words)}")
    if write:
        with open(f"word_list_{file}.txt", "w") as out_f:
            for word in words:
                print(word, file=out_f)
        create_data(file)
    else:
        return create_data(np.array(words), step)


def one_step_greedy(n: int = 0, disp: int = 10, data: dict = None) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in a single step.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and entropy in descending order."""

    if data is None:
        with open(f"pmfs_{n}.json", "r") as f:
            pmfs = json.load(f)
    else:
        pmfs = data

    entropy_list = []
    for guess, pmf in pmfs.items():
        entropy_list.append((guess, entropy(pmf.values())))

    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def two_step_greedy(n: int = 0, disp: int = 10, data: tuple[dict, dict] = None) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in two steps.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and sum of entropy of two steps in descending order."""
    # start_time = time()
    if data is None:
        with open(f"pmfs_{n}.json", "r") as f1, open(f"input_mass_function_{n}.json", "r") as f2:
            pmfs = json.load(f1)
            mass_func = json.load(f2)
    else:
        pmfs, mass_func = data

    entropy_list = []
    total = min(200, len(pmfs))  # maximum number of top words in one_step greedy to speed up the program.
    count = 1
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
        # count = timer(start_time, count, total)

    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def create_greedy() -> None:
    """Store the initial result of one_step_greedy and two_step_greedy in the corresponding JSON file
    to speed up the program."""
    # create_data()
    one_step = one_step_greedy(disp=-1)
    create_json("one_step_entropy", dict(one_step))
    print("one step done.")

    two_step = two_step_greedy(disp=-1)
    create_json("two_step_entropy", dict(two_step))
    print("two step done.")


def bot(step=1) -> None:
    """A bot that suggests best guess words in a game."""
    num = ["one", "two"]
    with open(f"{num[step - 1]}_step_entropy.json", "r") as f:
        entropy_list = list(json.load(f).items())
        print(entropy_list[:10])

    for i in range(1, 7):
        guess = input("Please input your guess:")
        if len(guess) != 5 or guess.isdigit():
            guess = entropy_list[0][0]
            print("Get invalid or empty input. Use default input instead.")
        pattern = input("please input the returned pattern:")
        if False:
            plot_pmf(guess, pattern, i - 1)
        if pattern == '22222':
            for j in range(1, i):
                del_data(j)
            break

        eliminate(guess, pattern, i)
        if step == 2:
            entropy_list = two_step_greedy(i)

            # if two_step_greedy() produced identical entropy for all guesses, it means that the number of feasible
            # guesses are small game, thus will finish within 2 steps. If so, one_steps_greedy() should be used instead.
            all_same = True
            first_entropy = entropy_list[0][1]
            for guess, value in entropy_list:
                if first_entropy - value < 1e-6:
                    continue
                else:
                    all_same = False
                    break
            if all_same:
                entropy_list = one_step_greedy(i)
                print("Using one-step greedy algorithm.")
            else:
                print("Using two-step greedy algorithm.")

        elif step == 1:
            entropy_list = one_step_greedy(i)

        print(entropy_list)

    print("fin.")


def simulator(step=1) -> None:
    """Simulate playing the game using wordle bot. Iterate over all possible answers.
    Return the average number of attempts using the algorithm."""
    start_time = time()

    with open("past_ans.txt", "r") as f:
        word_list = f.read().split()

    with open(f"two_step_entropy.json", "r") as f:
        start_word_list = list(json.load(f).items())

    total_total_word = len(word_list) * len(start_word_list)
    total_word = len(word_list)
    # always use words from two_step_entropy.json algo
    # with open(f"one_step_entropy.json", "r") as f:
    #    start_word = list(json.load(f).items())[0][0]

    with open("input_mass_function_0.json", "r") as f:
        base_mass_func = json.load(f)

    performance = []
    progress = 1
    for start_word, _ in start_word_list:
        total_count = 0
        for answer in word_list:
            i = 0
            mass_func = base_mass_func
            guess = start_word
            while True:
                i += 1
                pattern = check_word(guess, answer)

                if pattern == '22222':
                    print(i)
                    break

                # pass file data as parameter to avoid file I/O delay
                pmfs, mass_func = eliminate(guess, pattern, write=False, file=mass_func, step=2)
                if step == 2:
                    entropy_list = two_step_greedy(i, disp=6, data=(pmfs, mass_func))

                    # print(guess, answer, pattern)
                    # print(entropy_list)

                    all_same = True
                    first_entropy = entropy_list[0][1]
                    for guess, value in entropy_list:
                        if first_entropy - value < 1e-6:
                            continue
                        else:
                            all_same = False
                            break
                    if all_same:
                        entropy_list = one_step_greedy(i, disp=1, data=pmfs)

                        print("Using one-step greedy algorithm.")
                    else:
                        print("Using two-step greedy algorithm.")

                elif step == 1:
                    entropy_list = one_step_greedy(i, disp=1, data=pmfs)
                    print("Using one-step greedy algorithm.")

                guess = entropy_list[0][0]
            progress = timer(start_time, progress, total_total_word)
            total_count += i
        print(f"avg attempt:{total_count / total_word}")
        performance.append((start_word ,total_count / total_word))
    performance.sort(key=lambda x: x[-1], reverse=True)
    print(performance)


if __name__ == "__main__":
    simulator(1)
