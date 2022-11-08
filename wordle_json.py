import json
from wordle import check_word, iternary
import os


def create_map(file=0) -> None:
    """Create a 2D JSON file that the first key is guess, second key is answer,
    stored value is a 5-digit ternary string."""
    with open(f"word_list_{file}.txt", "r") as in_f:
        with open(f"input_answer_map_{file}.json", "w") as out_f:
            result = dict()
            word_list = in_f.read().split()
            for guess in word_list:
                result[guess] = dict()
                for answer in word_list:
                    result[guess][answer] = check_word(guess, answer)
            json.dump(result, out_f, indent=4)
            print(f"File input_answer_map_{file}.json created.")


def create_mf(file=0) -> None:
    """Create a 2D JSON file that the first key is guess, second key is a 5-digit ternary string,
    stored value is a list of possible answers"""
    with open(f"input_answer_map_{file}.json", "r") as in_f:
        with open(f"input_mass_function_{file}.json", "w") as out_f:
            result = dict()
            in_ans_map = json.load(in_f)
            for guess, answers in in_ans_map.items():
                result[guess] = dict()
                for answer, value in answers.items():
                    if value not in result[guess].keys():
                        result[guess][value] = [answer]
                    else:
                        result[guess][value].append(answer)
            for each, answers in result.items():
                result[each] = dict(sorted(answers.items(), key=lambda x: iternary(x[0])))
            json.dump(result, out_f, indent=4)
            print(f"File input_mass_function_{file}.json created.")


def create_pmf(file=0) -> None:
    """Create a 2D JSON file as a lists of PMF of random variable X given the guess, where X is the possible output
    patterns encoded as a 5-digit ternary string. The first key is guess, the second key is the 5-digit ternary number,
    the output is the probability"""
    with open(f"input_mass_function_{file}.json", "r") as in_f:
        with open(f"pmfs_{file}.json", "w") as out_f:

            result = dict()
            mass_func = json.load(in_f)
            size = len(mass_func)
            for guess, patterns in mass_func.items():
                result[guess] = dict()
                for pattern, answers in patterns.items():
                    result[guess][pattern] = len(answers) / size
            json.dump(result, out_f, indent=4)
            print(f"File pmfs_{file}.json created.")


def create_jsons(file=0) -> None:
    create_map(file)
    create_mf(file)
    create_pmf(file)


def del_map(file=0) -> None:
    """Delete input_answer_map.json"""
    os.remove(f"input_answer_map_{file}.json")


def del_mf(file=0) -> None:
    """Delete input_mass_function.json"""
    os.remove(f"input_mass_function_{file}.json")


def del_pmfs(file=0) -> None:
    """Delete pmfs.json"""
    os.remove(f"pmfs_{file}.json")


if __name__ == "__main__":
    create_map()
    create_mf()
    # del_map()
    # del_mf()
    create_pmf()
    pass
