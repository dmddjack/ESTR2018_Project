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


'''
class WordleManager(BaseManager):
    """Discarded."""
    pass


class WordleProxy(NamespaceProxy):
    """Discarded."""
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', '__mul__')

    def __mul__(self, other):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('__mul__', args=(other,))


def blockshaped(arr, nrows, ncols):
    """
    ***Discarded.***
    Return an array of shape (nrows, ncols, n, m) where
    n * nrows, m * ncols = arr.shape.
    This should be a view of the original array.
    """
    h, w = arr.shape
    n, m = h // nrows, w // ncols
    return arr.reshape(nrows, n, ncols, m).swapaxes(1, 2)


def do_dot(a, b, out):
    """Discarded."""
    # np.dot(a, b, out)  # does not work. maybe because out is not C-contiguous?
    out[:] = np.matmul(a, b)  # less efficient because the output is stored in a temporary array?
    #print("hi")


def pardot(a, b, nblocks, mblocks, dot_func=do_dot):
    """
    ***Discarded.***
    Return the matrix product a * b.
    The product is split into nblocks * mblocks partitions that are performed
    in parallel threads.
    """
    n_jobs = nblocks * mblocks
    print('running {} jobs in parallel'.format(n_jobs))

    out = np.empty((a.shape[0], b.shape[1]), dtype=np.ubyte)

    out_blocks = blockshaped(out, nblocks, mblocks)
    a_blocks = blockshaped(a, nblocks, 1)
    b_blocks = blockshaped(b, 1, mblocks)

    processes = []
    for i in range(nblocks):
        for j in range(mblocks):
            pr = mp.Process(target=dot_func,
                            args=(a_blocks[i, 0, :, :], b_blocks[0, j, :, :], out_blocks[i, j, :, :]))
            pr.start()
            processes.append(pr)

    for th in processes:
        th.join()
        print(f"th {th} done")
    print("all th done")
    result = np.empty((a.shape[0], b.shape[1]), dtype="<U5")
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            result[i, j] = ternary(out[i, j])
    return result
'''


def create_json(file_name: str = None, data: dict = None) -> None:
    """Create a JSON file with the given file name and dict data."""
    if file_name is not None:
        with open(f"{file_name}.json", "w") as f:
            json.dump(data, f, indent=4)


def create_map(file=0, debug=False) -> None | pd.DataFrame:
    """Create a 2D JSON file that the first key is guess, second key is answer,
    stored value is a 5-digit ternary string."""

    def find_map(data) -> pd.DataFrame:
        word_index = np.matrix([Wordle(i, data) for i in range(len(data))], dtype=object)
        result = pd.DataFrame(np.array(word_index.T @ word_index, dtype="<U5"), data, data, dtype="<U5")
        return result

    if isinstance(file, int):
        with open(f"word_list_{file}.txt", "r") as in_f:
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
        with open(f"input_answer_map_{file}.json", "r") as in_f:
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
        with open(f"input_mass_function_{file}.json", "r") as in_f:
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
    with open("word_list_0.txt", "r") as in_f, open("test_result.json", "w") as out_f:
        data = in_f.read().split()
        pmfs, mass_func = create_data(np.array(data), step=2, debug=True)
        result = {"pmfs": pmfs, "mass_func": mass_func}
        json.dump(result, out_f, indent=4)
    print("Done.")


def del_map(file=0) -> None:
    """Delete input_answer_map_{file}.json"""
    try:
        os.remove(f"input_answer_map_{file}.json")
    except OSError:
        pass


def del_mf(file=0) -> None:
    """Delete input_mass_function_{file}.json"""
    try:
        os.remove(f"input_mass_function_{file}.json")
    except OSError:
        pass


def del_pmfs(file=0) -> None:
    """Delete pmfs_{file}.json"""
    try:
        os.remove(f"pmfs_{file}.json")
    except OSError:
        pass


def del_word_lists(file=None) -> None:
    """Delete word_list_{file}.json"""
    if file != 0:
        try:
            os.remove(f"word_list_{file}.txt")
        except OSError:
            pass


def del_data(file=None, max=6) -> None:
    if file is not None:
        if file == -2:  # hard delete
            for i in range(0, max + 1):
                del_data(i)
        if file == -1:  # soft delete
            for i in range(1, max + 1):
                del_data(i)
        else:
            del_map(file)
            del_mf(file)
            del_pmfs(file)
            del_word_lists(file)


def update_ans_history():
    """Reformat the ans_list.txt file."""
    with open("past_ans.txt") as f:
        lst = f.read().split()

    with open("past_ans.txt", "w") as f:
        for each in lst:
            f.write(each.lower() + "\n")


if __name__ == "__main__":
    # create_data(debug=True)
    del_data(-1)
    create_data(debug=True)
