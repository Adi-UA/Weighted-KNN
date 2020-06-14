import pandas as pd
import pathlib


def read_CSV(filename):
    """
    This function reads a CSV with the given filename.

    Args:
        filename (str): The filename must contain the path to the file assuming this repository is the parent. So a file in the data folder inside the directory where this python file is located would simply need to be sent as "data/filename.extension"

    Returns:
        Dataframe: The pandas dataframe with the read data
    """

    filepath = str(pathlib.Path(__file__).parent.absolute()) + "/" + filename
    data = pd.read_csv(filepath)
    return data


def to_vector(df):
    """
    This function converts the given dataframe into a list of lists where each list is supposed to represent a vector.

    Args:
        df (Dataframe): The dataframe to change to vectors.

    Returns:
        list of lists: The list of vectors where each vector is a row of the input dataframe
    """

    vector_list = df.values.tolist()
    return vector_list


def random_train_test_split(df, train_frac, random_seed=None):
    """
    This function randomizes the dta based on the seed and then splits the dataframe into train and test sets which are changed to their list of vector representations.

    Args:
        df (Dataframe): The dataframe which is to be used to generate the train and test split.
        train_frac (int): The percentage in the range 0.0 to 1 that should be used for training.
        random_seed (int, optional): The seed for randomising. Defaults to None which means a seed is chosen at random.

    Returns:
        tuple: The list of lists representing the vectors in the train and test data frame respectively,
    """

    if random_seed is not None:
        df = df.sample(frac=1, random_state=random_seed)
    else:
        df = df.sample(frac=1)

    split_point = int(len(df.index) * train_frac)
    train = to_vector(df[:split_point])
    test = to_vector(df[split_point:])
    return train, test
