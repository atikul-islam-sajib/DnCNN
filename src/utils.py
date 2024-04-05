import sys
import os
import joblib
import torch
import torch.nn as nn

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


def device_init(device="mps"):
    """
    Initializes and returns the specified PyTorch device based on availability.

    This function checks the availability of the specified hardware acceleration (either MPS for Apple Silicon or CUDA for NVIDIA GPUs) and returns the appropriate device. If the requested device is not available, it falls back to CPU.

    Parameters:
        device (str): The type of device to initialize. Accepts 'mps' for Apple's Metal Performance Shaders, 'cuda' for NVIDIA GPUs, or any other value defaults to 'cpu'. Default is 'mps'.

    Returns:
        torch.device: The initialized PyTorch device.

    Example:
        >>> device = device_init('cuda')
        >>> print(device)
        device(type='cuda')  # This output depends on the availability of CUDA.

    Note:
        The 'mps' device option is specifically for use with Apple Silicon GPUs. Ensure your PyTorch installation supports MPS if using an Apple Silicon Mac.

    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def weight_init(m):
    """
    Applies custom weight initialization based on the layer type.

    This function initializes weights differently depending on the type of the module `m`. For convolutional layers (`Conv` in the class name), it applies He initialization (also known as Kaiming initialization). For batch normalization layers (`BatchNorm` in the class name), it initializes weights to a normal distribution with mean 1.0 and standard deviation 0.02, and biases to 0.

    Parameters:
        m (torch.nn.Module): The module to initialize.

    Note:
        This function is typically used as an argument to `apply` method of `nn.Module`. It checks the class name of `m` to determine the type of layer and applies the corresponding initialization. Unsupported layer types are not modified.

    Example:
        >>> model = MyModel()
        >>> model.apply(weight_init)

    The `nn.init.kaiming_normal_` method is used for convolutional layers to account for the rectifier nonlinearities, assuming they are followed by ReLU activations. The batch normalization layer weights are initialized from a normal distribution for better initial convergence.
    """
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(
            m.weight.data, mode="fan_out", nonlinearity="relu"
        )
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        torch.nn.init.constant_(m.bias.data, val=0.0)
