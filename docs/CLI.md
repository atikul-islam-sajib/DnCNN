## Command-Line Arguments

This tool provides a variety of options for configuring the training and testing of the DnCNN model. Below is a table summarizing the available command-line arguments.

| Argument           | Type  | Default           | Description                                                 |
| ------------------ | ----- | ----------------- | ----------------------------------------------------------- |
| `--image_path`     | str   | `./data/data.zip` | Path to the zip file containing the dataset.                |
| `--batch_size`     | int   | `16`              | Batch size for training and testing.                        |
| `--image_size`     | int   | `64`              | The size to which the images will be resized.               |
| `--split_ratio`    | float | `0.2`             | The ratio of the dataset to be used as the test set.        |
| `--epochs`         | int   | `100`             | Number of training epochs.                                  |
| `--lr`             | float | `1e-4`            | Learning rate for the optimizer.                            |
| `--device`         | str   | `mps`             | Computing device (`cuda`, `cpu`, `mps`).                    |
| `--display`        | bool  | `True`            | Whether to display training progress and metrics.           |
| `--beta1`          | float | `0.9`             | Beta1 hyperparameter for Adam optimizer.                    |
| `--train`          | flag  | N/A               | Flag to initiate model training.                            |
| `--test`           | flag  | N/A               | Flag to initiate model testing and visualization.           |
| `--adam`           | bool  | `True`            | Whether to use Adam optimizer.                              |
| `--SGD`            | bool  | `False`           | Whether to use Stochastic Gradient Descent (SGD) optimizer. |
| `--is_l1`          | bool  | `False`           | Whether to use L1 regularization.                           |
| `--is_l2`          | bool  | `False`           | Whether to use L2 regularization.                           |
| `--is_huber_loss`  | bool  | `False`           | Whether to use Huber loss as the training criterion.        |
| `--is_weight_clip` | bool  | `False`           | Whether to apply weight clipping.                           |

For running your model on different devices such as CPU, CUDA, and MPS (Metal Performance Shaders for Apple Silicon), it's essential to provide clear instructions tailored to each computing platform. Below is a markdown table that outlines how to specify the device type using the `--device` argument in your command line interface (CLI).

### Specifying Compute Devices

| Device Type | Argument        | Description                                  |
| ----------- | --------------- | -------------------------------------------- |
| CPU         | `--device cpu`  | Run the model on the CPU.                    |
| CUDA        | `--device cuda` | Run the model on NVIDIA CUDA-supported GPUs. |
| MPS         | `--device mps`  | Run the model on Apple Silicon GPUs.         |

### Example Usage for Different Devices

To initiate model training or testing on different devices, use the `--device` argument as shown in the examples below:

**Training on CPU:**

```bash
python cli.py --train --epochs 50 --batch_size 32 --lr 0.0001 --device cpu ..... # Based on your requirements
```

**Training on CUDA GPU:**

```bash
python cli.py --train --epochs 50 --batch_size 32 --lr 0.0001 --device cuda .....  # Based on your requirements
```

**Training on Apple Silicon GPU (MPS):**

```bash
python cli.py --train --epochs 50 --batch_size 32 --lr 0.0001 --device mps ......  # Based on your requirements
```

**Testing on CPU:**

```bash
python cli.py --test --device cpu
```

**Testing on CUDA GPU:**

```bash
python cli.py --test --device cuda
```

**Testing on Apple Silicon GPU (MPS):**

```bash
python cli.py --test --device mps
```
