import sys

sys.path.append("..")

import os

import absl.app
import absl.flags
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch  # type: ignore
import torchvision  # type: ignore

import utils.datasets as datasets
import utils.utils as utils


# User flags
absl.flags.DEFINE_string(
    "path_model",
    None,
    "Path of the trained model",
)
absl.flags.DEFINE_integer(
    "batch_size_test",
    3,
    "Inference batch size",
)
absl.flags.DEFINE_string(
    "dir_dataset",
    "../datasets/",
    "Directory where datasets are stored",
)

# Task 6 flags
absl.flags.DEFINE_integer(
    "num_wrong_images",
    20,
    "Number of incorrectly predicted test images to save",
)
absl.flags.DEFINE_integer(
    "min_index",
    20,
    "Ignore test images with an absolute index below this value",
)
absl.flags.DEFINE_string(
    "output_dir",
    "../images/task6_wrong_predictions",
    "Directory where Task 6 figures are saved",
)

absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS


def run(path: str, dataset_dir: str):
    """Generate memory images for incorrectly predicted test images.

    Each output figure includes:
      - The absolute index of the input in the test dataset
      - The true class
      - The predicted class
      - Memory samples with a positive influence on the prediction

    Args:
        path: Path to the trained model checkpoint.
        dataset_dir: Directory where datasets are stored.
    """

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print("Device: {}".format(device))

    # Load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint["modality"]

    if modality not in ["memory", "encoder_memory"]:
        raise ValueError(
            "Model modality must be 'memory' or 'encoder_memory', not {}".format(
                modality
            )
        )

    dataset_name = checkpoint["dataset_name"]

    model = utils.get_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        model_type=modality,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load data
    train_examples = checkpoint["train_examples"]

    if dataset_name in ["CIFAR10", "CINIC10"]:
        name_classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    else:
        name_classes = range(checkpoint["num_classes"])

    load_dataset = getattr(datasets, "get_" + dataset_name)
    undo_normalization = getattr(
        datasets,
        "undo_normalization_" + dataset_name,
    )

    batch_size_test = FLAGS.batch_size_test

    _, _, test_loader, mem_loader = load_dataset(
        dataset_dir,
        batch_size_train=50,
        batch_size_test=batch_size_test,
        batch_size_memory=100,
        size_train=train_examples,
    )

    memory_iter = iter(mem_loader)

    # Save Task 6 figures in their own directory.
    dir_save = FLAGS.output_dir
    os.makedirs(dir_save, exist_ok=True)

    def get_image(image, revert_norm=True):
        if revert_norm:
            image = undo_normalization(image)

        image = image.squeeze().cpu().detach().numpy()
        return np.transpose(image, (1, 2, 0))

    saved_count = 0
    batch_start_index = 0

    # test_loader is sequential, so the running offset gives the absolute
    # position of each input in the test dataset.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            print(
                "Scanning batch: {}/{}".format(
                    batch_idx + 1,
                    len(test_loader),
                ),
                end="\r",
            )

            try:
                memory, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(mem_loader)
                memory, _ = next(memory_iter)

            images = images.to(device)
            memory = memory.to(device)

            # Compute predictions and memory weights.
            outputs, memory_weights = model(
                images,
                memory,
                return_weights=True,
            )
            _, predictions = torch.max(outputs, 1)

            memory_values, memory_sorted_indices = torch.sort(
                memory_weights,
                descending=True,
            )

            for index_in_batch in range(len(images)):
                absolute_index = batch_start_index + index_in_batch

                true_index = int(labels[index_in_batch].item())
                predicted_index = int(
                    predictions[index_in_batch].item()
                )

                # Do not use inputs among the first 20 dataset entries.
                if absolute_index < FLAGS.min_index:
                    continue

                # Only generate figures for incorrect predictions.
                if predicted_index == true_index:
                    continue

                input_selected = images[index_in_batch].unsqueeze(0)

                # Select memory samples with a positive prediction weight.
                positive_memory_indices = memory_sorted_indices[
                    index_in_batch
                ][memory_values[index_in_batch] > 0]

                reduced_memory = undo_normalization(
                    memory[positive_memory_indices]
                )

                memory_grid = torchvision.utils.make_grid(
                    reduced_memory,
                    nrow=4,
                ).cpu().numpy()

                true_class = name_classes[true_index]
                predicted_class = name_classes[predicted_index]

                # Create one figure for this incorrect prediction.
                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(8, 8),
                    dpi=200,
                )

                axes[0].imshow(
                    (get_image(input_selected) * 255).astype(np.uint8),
                    interpolation="nearest",
                    aspect="equal",
                )
                axes[0].set_title(
                    "Absolute index: {}\n"
                    "True class: {} | Predicted class: {}".format(
                        absolute_index,
                        true_class,
                        predicted_class,
                    )
                )
                axes[0].axis("off")

                axes[1].imshow(
                    (
                        np.transpose(memory_grid, (1, 2, 0)) * 255
                    ).astype(np.uint8),
                    interpolation="nearest",
                    aspect="equal",
                )
                axes[1].set_title(
                    "Positively weighted memory samples"
                )
                axes[1].axis("off")

                fig.tight_layout()

                filename = (
                    "wrong_index_{}_true_{}_pred_{}.png".format(
                        absolute_index,
                        true_index,
                        predicted_index,
                    )
                )

                output_path = os.path.join(dir_save, filename)
                fig.savefig(output_path)
                plt.close(fig)

                saved_count += 1

                print(
                    "\nSaved {}/{}: {}".format(
                        saved_count,
                        FLAGS.num_wrong_images,
                        output_path,
                    )
                )

                if saved_count >= FLAGS.num_wrong_images:
                    break

            batch_start_index += len(images)

            if saved_count >= FLAGS.num_wrong_images:
                break

    print(
        "\nFinished. Saved {} incorrect-prediction figures to {}".format(
            saved_count,
            dir_save,
        )
    )

    if saved_count < FLAGS.num_wrong_images:
        print(
            "Warning: only {} qualifying predictions were found.".format(
                saved_count
            )
        )


def main(argv):
    del argv
    run(FLAGS.path_model, FLAGS.dir_dataset)


if __name__ == "__main__":
    absl.app.run(main)