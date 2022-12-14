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
        with open(f"./data/input_mass_function_{file - 1}.json", "r") as in_f:
            words = json.load(in_f)[guess][pattern]
    else:
        write = False
        words = file[guess][pattern]
    # print(f"data size: {len(words)}")
    # print(f"Number of remaining choices: {len(words)}")
    if write:
        with open(f"./data/word_list_{file}.txt", "w") as out_f:
            for word in words:
                print(word, file=out_f)
        create_data(file)
    else:
        return create_data(np.array(words), step)


def rearrange(entropy_data):
    """Fit the dataset to the feasible answers."""
    with open("./data/answer_list.txt", "r") as f:
        ans_list = f.read().split()

    greatest_entropy = entropy_data[0][1]
    pos = 0
    for index, data in enumerate(entropy_data):
        word, value = data
        if greatest_entropy - value < 1e-5:
            pos = index
        else:
            break

    for j in range(1, pos + 1):
        if entropy_data[j][0] in ans_list:
            entropy_data.insert(0, entropy_data.pop(j))


def one_step_greedy(n: int = 0, disp: int = 10, data: dict = None) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in a single step.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and entropy in descending order."""

    if data is None:
        with open(f"./data/pmfs_{n}.json", "r") as f:
            pmfs = json.load(f)
    else:
        pmfs = data

    entropy_list = []
    for guess, pmf in pmfs.items():
        entropy_list.append((guess, entropy(pmf.values())))

    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def two_step_greedy(n: int = 0, disp: int = 10, data: tuple[dict, dict] = None, debug=False) -> list[tuple]:
    """Use the greedy algorithm to find the maximum entropy and the corresponding word in two steps.
    disp indicates the number of items to be returned. the whole list is returned if disp == -1.
    n is the number of times of attempt. Return a list of words and sum of entropy of two steps in descending order."""
    if debug:
        start_time = time()
        count = 1
    if data is None:
        with open(f"./data/pmfs_{n}.json", "r") as f1, open(f"./data/input_mass_function_{n}.json", "r") as f2:
            pmfs = json.load(f1)
            mass_func = json.load(f2)
    else:
        pmfs, mass_func = data

    entropy_list = []
    total = min(200, len(pmfs))  # restrict maximum number of top words in one_step greedy to speed up the program.

    top_words = dict(one_step_greedy(n + 1, total, pmfs)).keys()
    for guess, pmf in pmfs.items():
        if guess not in top_words:
            continue
        sec_ord_entropy = 0  # define the entropy at the second step
        for pattern, p in pmf.items():  # find maximum entropy at the second step
            sub_pmfs = eliminate(guess, pattern, file=mass_func, write=False)
            sec_ord_entropy += p * one_step_greedy(n + 1, 1, sub_pmfs)[0][1]
            # weighted average of entropy
        entropy_list.append((guess, entropy(pmf.values()) + sec_ord_entropy))
        if debug:
            count = timer(start_time, count, total)

    entropy_list.sort(key=lambda x: x[-1], reverse=True)
    return entropy_list[:disp] if disp != -1 else entropy_list


def create_greedy() -> None:
    """Store the initial result of one_step_greedy and two_step_greedy in the corresponding JSON file
    to speed up the program."""
    # create_data()
    one_step = one_step_greedy(disp=-1)
    create_json("one_step_entropy", dict(one_step))
    print("one step done.")

    two_step = two_step_greedy(disp=-1, debug=True)
    create_json("two_step_entropy", dict(two_step))
    print("two step done.")


def bot(step=1, plot=False) -> None:
    """A bot that suggests best guess words in a game."""
    with open(f"./data/two_step_entropy.json", "r") as f:  # always use two step data to choose start word
        entropy_list = list(json.load(f).items())
        print(entropy_list[:10])

    for i in range(1, 10):
        if len(entropy_list) == 1:
            del_data(file=-1, _max=i)
            print(f"number of attempt: {i}")
            break
        guess = input("Please input your guess:")
        if len(guess) != 5 or guess.isdigit():
            guess = entropy_list[0][0]
            print(f"Get invalid or empty input. Use default input '{guess}' instead.")
        pattern = input("please input the returned pattern:")
        if pattern == '22222':
            del_data(file=-1, _max=i)
            print(f"number of attempt: {i}")
            break
        if plot:
            plot_pmf(guess, pattern, i - 1)

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

        rearrange(entropy_list)
        print(entropy_list)

    print("Finished")


def simulator(step=1) -> None:
    """Simulate playing the game using wordle bot. Iterate over all possible answers.
    Return the average number of attempts using the algorithm."""
    start_time = time()
    with open("./data/past_ans.txt", "r") as f:
        word_list = f.read().split()

    with open("./data/two_step_entropy.json", "r") as f:  # always use words from two_step_entropy.json
        start_word_list = list(json.load(f).items())[:10]

    total_total_word = len(word_list) * len(start_word_list)
    total_word = len(word_list)

    with open("./data/input_mass_function_0.json", "r") as f:
        base_mass_func = json.load(f)

    performance = []
    progress = 1
    for start_word, _ in start_word_list:
        attempt_stat = [0] * 8
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
                    entropy_list = two_step_greedy(i, disp=10, data=(pmfs, mass_func))

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
                        entropy_list = one_step_greedy(i, disp=10, data=pmfs)

                        print("Using one-step greedy algorithm.")
                    else:
                        print("Using two-step greedy algorithm.")

                elif step == 1:
                    entropy_list = one_step_greedy(i, disp=10, data=pmfs)
                    print("Using one-step greedy algorithm.")
                else:
                    raise ValueError("Invalid step size!")

                rearrange(entropy_list)

                guess = entropy_list[0][0]
            progress = timer(start_time, progress, total_total_word)
            total_count += i
            attempt_stat[min(i, 7)] += 1
        print(f"avg attempt:{total_count / total_word}")
        print(f"attempt distribution: {attempt_stat[1:]}")
        performance.append((start_word, total_count / total_word))
    performance.sort(key=lambda x: x[-1], reverse=False)
    print(performance)


if __name__ == "__main__":
    simulator(1)
