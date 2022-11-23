import json
import os

import numpy as np
import pandas as pd

from wordle import check_word, iternary


class Wordle(int):
    def __new__(cls, value, words):
        return super(Wordle, cls).__new__(cls, value)

    def __init__(self, value, words):
        super().__init__()
        self.value = int(value)
        self.word = words[self.value]

    def __mul__(self, other):
        return check_word(self.word, other.word)


def create_json(file_name: str = None, data: dict = None) -> None:
    """Create a JSON file with the given file name and dict data."""
    if file_name is not None:
        with open(f"./data/{file_name}.json", "w") as f:
            json.dump(data, f, indent=4)


def create_map(file=0, debug=False) -> None | pd.DataFrame:
    """Create a 2D JSON file that the first key is guess, second key is answer,
    stored value is a 5-digit ternary string."""

    def find_map(data) -> pd.DataFrame:
        word_index = np.matrix([Wordle(i, data) for i in range(len(data))], dtype=object)
        result = pd.DataFrame(np.array(word_index.T @ word_index, dtype="<U5"), data, data, dtype="<U5")
        return result

    if isinstance(file, int):
        with open(f"./data/word_list_{file}.txt", "r") as in_f:
            words = np.array(in_f.read().split())
            result = find_map(words).to_dict("index")
            create_json(f"input_answer_map_{file}", result)
            if debug:
                print(f"File input_answer_map_{file}.json created.")
    elif isinstance(file, np.ndarray):
        """
        WordleManager.register("Wordle", Wordle, WordleProxy)
        with WordleManager() as manager:
            word_index = np.ndarray((1, len(file)), dtype=object)
            for i in range(len(file)):
                word_index[0, i] = manager.Wordle(i, file)
            result = np.array(pardot(word_index.reshape(len(file), 1),
                                     word_index, 10 , 10), dtype="<U5")
        """
        result = find_map(file)
        if debug:
            print(f"Dict map created.")

        return result


def create_mf(file: int | pd.DataFrame = 0, debug=False) -> None | dict:
    """Create a 2D JSON file that the first key is guess, second key is a 5-digit ternary string,
    stored value is a list of possible answers."""

    def find_mass_function(data: dict) -> dict:
        result = dict()
        for guess, answers in data.items():
            result[guess] = dict()
            for answer, value in answers.items():
                if value not in result[guess].keys():
                    result[guess][value] = [answer]
                else:
                    result[guess][value].append(answer)
        for each, answers in result.items():
            result[each] = dict(sorted(answers.items(), key=lambda x: iternary(x[0])))
        return result

    if isinstance(file, int):
        with open(f"./data/input_answer_map_{file}.json", "r") as in_f:
            in_ans_map = json.load(in_f)
            result = find_mass_function(in_ans_map)
            create_json(f"input_mass_function_{file}", result)
            if debug:
                print(f"File input_mass_function_{file}.json created.")
    elif isinstance(file, pd.DataFrame):
        file = file.to_dict("index")
        result = find_mass_function(file)
        if debug:
            print(f"Dict mass function created.")
        return result


def create_pmfs(file: int | pd.DataFrame = 0, debug=False) -> None | dict:
    """Create a 2D JSON file as a lists of PMF of random variable X given the guess, where X is the possible output
    patterns encoded as a 5-digit ternary string. The first key is guess, the second key is the 5-digit ternary number,
    the output is the probability."""

    def find_pmfs(data: dict) -> dict:
        result = dict()
        size = len(data)
        for guess, patterns in data.items():
            result[guess] = dict()
            for pattern, answers in patterns.items():
                result[guess][pattern] = len(answers) / size
        return result

    if isinstance(file, int):
        with open(f"./data/input_mass_function_{file}.json", "r") as in_f:
            mass_func = json.load(in_f)
            result = find_pmfs(mass_func)
            create_json(f"pmfs_{file}", result)
            if debug:
                print(f"File pmfs_{file}.json created.")
    elif isinstance(file, pd.DataFrame):
        file = file.transpose()
        result = {}
        for col in file:
            result[col] = file[col].value_counts(normalize=True, sort=False).to_dict()
        if debug:
            print(f"Dict pmf created.")

        return result


def create_data(file: int | np.ndarray = 0, step=1, debug=False) -> None | dict | tuple[dict, dict]:
    """Create all the required data for wordle_bot.py in a row."""
    if isinstance(file, int):
        create_map(file, debug)
        create_mf(file, debug)
        create_pmfs(file, debug)
    elif isinstance(file, np.ndarray) and step == 1:
        pmfs = create_pmfs(create_map(file), debug)
        return pmfs
    elif isinstance(file, np.ndarray) and step == 2:
        in_out_map = create_map(file, debug)
        pmfs, mass_func = create_pmfs(in_out_map, debug), create_mf(in_out_map, debug)
        return pmfs, mass_func


def test_create_data():
    """For debugging only."""
    with open("./data/word_list_0.txt", "r") as in_f, open("./data/test_result.json", "w") as out_f:
        data = in_f.read().split()
        pmfs, mass_func = create_data(np.array(data), step=2, debug=True)
        result = {"pmfs": pmfs, "mass_func": mass_func}
        json.dump(result, out_f, indent=4)
    print("Done.")


def del_map(file=0) -> None:
    """Delete input_answer_map_{file}.json"""
    try:
        os.remove(f"./data/input_answer_map_{file}.json")
    except OSError:
        pass


def del_mf(file=0) -> None:
    """Delete input_mass_function_{file}.json"""
    try:
        os.remove(f"./data/input_mass_function_{file}.json")
    except OSError:
        pass


def del_pmfs(file=0) -> None:
    """Delete pmfs_{file}.json"""
    try:
        os.remove(f"./data/pmfs_{file}.json")
    except OSError:
        pass


def del_word_lists(file=None) -> None:
    """Delete word_list_{file}.json"""
    if file != 0:
        try:
            os.remove(f"./data/word_list_{file}.txt")
        except OSError:
            pass


def del_data(file=None, _max=6) -> None:
    if file is not None:
        if file == -2:  # hard delete, back to the state before initialization
            for i in range(0, _max + 1):
                del_data(i)
            try:
                os.remove("./data/one_step_entropy.json")
            except OSError:
                pass
            try:
                os.remove("./data/two_step_entropy.json")
            except OSError:
                pass

        if file == -1:  # soft delete
            for i in range(1, _max + 1):
                del_data(i)

        else:
            del_map(file)
            del_mf(file)
            del_pmfs(file)
            del_word_lists(file)


def update_ans_history():
    """Reformat the ans_list.txt file."""
    with open("./data/past_ans.txt") as f:
        lst = f.read().split()

    with open("./data/past_ans.txt", "w") as f:
        for each in lst:
            f.write(each.lower() + "\n")
    # last update: 23 Nov 2022


if __name__ == "__main__":
    # create_data(debug=True)
    del_data(-1)
    create_data(debug=True)
