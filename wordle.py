# This is a simplified Wordle game
import json
import math
import random

import matplotlib.pyplot as plt
import numpy as np


def ternary(n: int) -> str:
    """Turn a decimal into a ternary string"""
    result = []
    while True:
        n, remainder = divmod(n, 3)  # quotient and the remainder
        result.append(str(remainder))  # store the reminders in a list
        if n == 0:
            return ("".join(result[::-1])).zfill(5)  # return a ternary string


def iternary(s: str) -> int:
    """Turn a ternary string into a decimal number."""
    return int(s, base=3)


def check_word(guess: str, answer: str) -> str:
    answer = list(answer)
    result = ['0'] * 5
    for i, each in enumerate(guess):
        if each == answer[i]:
            result[i] = '2'
            answer[i] = None
    for i, each in enumerate(guess):
        if result[i] == '0' and each in answer:
            result[i] = '1'
            answer.remove(each)
    return "".join(result)


def generate_answer(seed: int, is_answer=True) -> str | list:
    """Generate a random answer from the word list if is_answer == True
    and return the answer list if is_answer == False."""
    random.seed(seed)
    with open('./data/word_list_0.txt', 'r') as f:
        allowed_word = f.read().split()
    with open('./data/answer_list.txt', 'r') as f:
        ans_list = f.read().split()
    answer = random.sample(ans_list, 1)[0]  # get a random answer from the word list
    f.close()
    return answer if is_answer else allowed_word


def plot_attempt_distribution() -> None:
    player = [0.02, 5.58, 22.35, 32.99, 24.08, 11.94, 3.04]
    bot = [0.00, 4.41, 44.64, 41.38, 8.04, 0.96, 0.57]
    x = ["1", "2", "3", "4", "5", "6", ">7"]
    x_axis = np.arange(7)

    plt.figure(figsize=(9, 6), dpi=300)
    plt.bar_label(plt.bar(x_axis - .2, player, color="#c8b653", width=.4, label="Global Player"))
    plt.bar_label(plt.bar(x_axis + .2, bot, color="#6ca965", width=.4, label="Wordle Bot"))
    plt.xticks(x_axis, x)
    plt.xlabel("Number of Attempts", fontsize=12)
    plt.ylabel("(%)", rotation=0, loc="top", fontsize=13)
    plt.title("Attempt Distribution of Global Player and Wordle Bot in Wordle Game", fontsize=14)
    plt.legend()
    plt.savefig("data/attempt_distribution.png", dpi=200)
    plt.show()


def plot_pmf(guess: str, pattern: str, i: int = 0) -> None:
    """plot pmf for guess word"""
    with open(f"./data/pmfs_{i}.json", "r") as f:
        pmfs = json.load(f)

    pmf = pmfs[guess]
    sorted_item = sorted(pmf.items(), key=lambda x: x[1], reverse=True)

    patterns = [item[0] for item in sorted_item]
    p_desc = [item[1] for item in sorted_item]
    infos = [math.log2(1 / p) if p != 0 else 0 for p in p_desc]

    index = patterns.index(pattern)
    p = p_desc[index]
    info = math.log2(1 / p)

    pattern = [str(i) for i in pattern]
    pattern_str = ','.join(pattern)
    length = len(p_desc)
    plt.subplot(211)
    plt.bar([i for i in range(len(p_desc))], p_desc, color='slateblue')
    plt.title('PMF')
    plt.bar(index, p, color='cadetblue')
    plt.xlim([-1, length + 1])
    plt.ylim([0, p_desc[0] * 1.2])
    plt.xticks(np.arange(0, len(patterns), 5), patterns[0:len(patterns):5], rotation=80, size=6)
    plt.annotate('P({})={:.6f}'.format(pattern_str, p), (index, p), (index - length / 10., p + (p_desc[0] - p) * 0.2),
                 weight='light', color='cadetblue', fontsize=6)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    plt.subplot(212)
    plt.bar([i for i in range(len(infos))], infos, color='lightcoral')
    plt.title('Information Spectrum')
    plt.bar(index, info, color='cadetblue')
    plt.xlim([-1, length + 1])
    plt.ylim([0, infos[-1] * 1.2])
    plt.xticks(np.arange(0, len(patterns), 5), patterns[0:len(patterns):5], rotation=80, size=6)
    plt.annotate('P({})={:.6f}'.format(pattern_str, info), (index, info),
                 (index - length / 10., info + (infos[-1] - info) * 0.2), weight='light', color='cadetblue', fontsize=6)
    # plt.savefig("test.png", dpi=300)
    plt.show()
    print('possibility:', p)
    print('get information:', info)


def main(seed=None) -> None:
    """The flow of the game"""
    from wordle_json import del_data

    attempt = 1
    opportunity = eval(input('Please enter the maximum number of attempts: '))
    if seed is None:
        seed = eval(input('Please enter a seed: '))
    answer = generate_answer(seed, True)
    while attempt <= opportunity or opportunity == -1:  # if opportunity == -1, the number of attempts is unlimited
        while True:  # make sure that the user's guess is in the word list
            guess = str(input('Please enter your guess: '))
            if guess in generate_answer(seed, is_answer=False):
                break
            else:
                print('Not in the answer list. Please enter again. ')
        print(f"{check_word(guess, answer):>30}")

        if check_word(guess, answer) == '22222':
            print(f'Correct! Number of attempt: {attempt}.')
            break
        else:
            attempt += 1
    else:
        print(f'Game over! The answer is "{answer}".')
        """ if the user fails to figure out the answer within the maximum number of attempts,
        print out the correct answer. """

    del_data(file=-1, _max=attempt)


if __name__ == '__main__':
    main()
