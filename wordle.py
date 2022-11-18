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


def check_word_old(guess: str, answer: str) -> str:
    """Check the guess and return the matching pattern as ternary number."""
    result = ''

    def redundancy(guess: str, answer: str, func: bool = True) -> bool | str:
        flag = 0
        for word in guess:
            if word in answer and guess.count(word) > answer.count(word):
                flag = 1
                alpha = word
                break
            else:
                continue
        if func:
            return True if flag == 1 else False
        if not func:
            return alpha

    for char, word in zip(answer, guess):
        if word in answer and word in char:
            result += '2'
        elif word in answer:
            result += '1'
        else:
            result += '0'

    if not redundancy(guess, answer, func=True):
        return result
    else:
        alpha = redundancy(guess, answer, func=False)
        index1 = guess.find(alpha)
        index2 = guess.find(alpha, index1 + 1)
        if index2 == answer.find(alpha):
            result = result[:index1] + '0' + result[index1 + 1:]
        else:
            result = result[:index2] + '0' + result[index2 + 1:]
        return result


def generate_answer(seed: int, is_answer=True) -> str | list:
    """Generate a random answer from the word list if is_answer == True
    and return the answer list if is_answer == False."""
    random.seed(seed)
    with open('answer_list.txt', 'r') as f:
        ans_list = []
        lines = f.readlines()
        for line in lines:
            ans_list.append(line[:5])  # exclude the "newline"
    answer = random.sample(ans_list, 1)[0]  # get a random answer from the word list
    f.close()
    return answer if is_answer else ans_list


def plot_pmf(guess: str, pattern: str, i: int = 0) -> None:
    """plot pmf for guess word
    """
    with open(f"pmfs_{i}.json", "r") as f:
        pmfs = json.load(f)

    pmf = pmfs[guess]
    sorted_item = sorted(pmf.items(),key=lambda x :x[1], reverse=True)

    patterns = [item[0] for item in sorted_item]
    p_desc = [item[1] for item in sorted_item]
    infos = [math.log2(1/p) if p != 0 else 0 for p in p_desc]

    index = patterns.index(pattern)
    p= p_desc[index]
    info = math.log2(1/p)

    pattern =[str(i) for i in pattern]
    pattern_str =','.join(pattern)
    length = len(p_desc)
    plt.subplot(211)
    plt.bar([i for i in range(len(p_desc))], p_desc, color='slateblue')
    plt.title('PMF')
    plt.bar(index,p,color='cadetblue')
    plt.xlim([-1, length + 1])
    plt.ylim([0, p_desc[0] * 1.2])
    plt.xticks(np.arange(0,len(patterns),5),patterns[0:len(patterns):5],rotation=80,size=6)
    plt.annotate('P({})={:.6f}'.format(pattern_str, p), (index, p), (index - length / 10., p + (p_desc[0] - p) * 0.2), weight='light', color='cadetblue', fontsize=6)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

    plt.subplot(212)
    plt.bar([i for i in range(len(infos))], infos, color='lightcoral')
    plt.title('Information Spectrum')
    plt.bar(index,info,color='cadetblue')
    plt.xlim([-1, length + 1])
    plt.ylim([0, infos[-1] * 1.2])
    plt.xticks(np.arange(0,len(patterns),5),patterns[0:len(patterns):5],rotation=80,size=6)
    plt.annotate('P({})={:.6f}'.format(pattern_str, info), (index, info), (index - length / 10., info + (infos[-1] - info) * 0.2), weight='light', color='cadetblue', fontsize=6)
    plt.show()
    print('possibility:', p)
    print('get information:', info)


def main() -> None:
    """The flow of the game"""
    from wordle_bot import eliminate
    from wordle_json import del_data

    attempt = 1
    opportunity = eval(input('Please enter the maximum number of attempts: '))
    seed = eval(input('Please enter a seed: '))
    answer = generate_answer(seed, True)
    while attempt <= opportunity or opportunity == -1:  # if opportunity == -1, the number of attempts is unlimited
        while True:  # make sure that the user's guess is in the word list
            guess = str(input('Please enter your guess: '))
            if guess in generate_answer(seed, is_answer=False):
                plot_pmf(guess, check_word(guess, answer), attempt-1)
                break
            else:
                print('Not in the answer list. Please enter again. ')
        print(f"{check_word(guess, answer):>30}")
        eliminate(guess, check_word(guess, answer), attempt)
        if check_word(guess, answer) == '22222':
            print(f'Correct! Number of attempt: {attempt}.')
            break
        else:
            attempt += 1
    else:
        print(f'Game over! The answer is "{answer}".')
        """ if the user fails to figure out the answer within the maximum number of attempts,
        print out the correct answer. """

    del_data(file=-1, max=attempt)


if __name__ == '__main__':
    main()