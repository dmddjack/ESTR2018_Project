# This is a simplified Wordle game
import random


def ternary(n: int) -> str:
    """Turn a decimal into a ternary string"""
    result = []
    while True:
        n, remainder = divmod(n, 3)  # quotient and the remainder
        result.append(str(remainder))  # store the reminders in a list
        if n == 0:
            return "".join(result[::-1])  # return a ternary string


def iternary(s: str) -> int:
    """Turn a ternary string into a decimal number."""
    return int(s, base=3)


def check_word(guess: str, answer: str) -> str:
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
    with open('word_list_0.txt', 'r') as f:
        ans_list = []
        lines = f.readlines()
        for line in lines:
            ans_list.append(line[:5])  # exclude the "newline"
    answer = random.sample(ans_list, 1)[0]  # get a random answer from the word list
    f.close()
    return answer if is_answer else ans_list


def main() -> None:
    """The flow of the game"""
    attempt = 1
    opportunity = eval(input('Please enter the maximum number of attempts: '))
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


if __name__ == '__main__':
    main()
