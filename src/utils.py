import sys
import os
import joblib

sys.path.append("src/")

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def dump(value, filename):
    """
    Serializes and saves a Python object to a file using joblib.

    Parameters:
        value (any): The Python object to serialize.
        filename (str): The path to the file where the object should be saved.

    Raises:
        Exception: If either `value` or `filename` is None, indicating missing required arguments.
    """
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)

    else:
        raise Exception("Please provide a valid path".capitalize())


def load(filename):
    """
    Loads a Python object from a file serialized with joblib.

    Parameters:
        filename (str): The path to the file from which the object should be loaded.

    Returns:
        any: The deserialized Python object.

    Raises:
        Exception: If `filename` is None, indicating a missing required argument.
    """
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise Exception("Please provide a valid path".capitalize())


def clean():
    """
    Removes the contents of directories specified by `RAW_DATA_PATH` and `PROCESSED_DATA_PATH`.

    The function deletes the first directory found in `RAW_DATA_PATH` and all files within `PROCESSED_DATA_PATH`.
    It is intended to prepare the environment for a new dataset by clearing out old data.

    Raises:
        Exception: If either `RAW_DATA_PATH` or `PROCESSED_DATA_PATH` does not exist or points to an invalid path.
    """
    if os.path.exists(RAW_DATA_PATH):
        directory = os.listdir(RAW_DATA_PATH)

        if os.path.isdir(RAW_DATA_PATH):
            os.system("rm -rf {}".format(os.path.join(RAW_DATA_PATH, directory[0])))
            print("done")

        else:
            raise Exception("Please provide a valid path".capitalize())
    else:
        raise Exception("Please provide a valid path".capitalize())

    if os.path.exists(PROCESSED_DATA_PATH):

        for file in os.listdir(PROCESSED_DATA_PATH):
            os.remove(os.path.join(PROCESSED_DATA_PATH, file))

    else:
        raise Exception("Please provide a valid path".capitalize())
