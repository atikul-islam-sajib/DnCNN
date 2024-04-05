import sys
import os
import joblib

sys.path.append("src/")

from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def dump(value, filename):
    if value is not None and filename is not None:
        joblib.dump(value=value, filename=filename)

    else:
        raise Exception("Please provide a valid path".capitalize())


def load(filename):
    if filename is not None:
        return joblib.load(filename=filename)
    else:
        raise Exception("Please provide a valid path".capitalize())


def clean():
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
