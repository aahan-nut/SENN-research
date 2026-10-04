import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_memory_impact import (
    build_correct_class_memory,
    build_random_memory,
    group_training_indices_by_class,
    load_model,
    load_targets,
    predict_with_memory,
)

import utils.datasets as datasets


def load_results(csv_path):
    """Load a results CSV and index its rows by absolute index."""

    results = {}

    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            results[int(row["absolute_index"])] = row

    return results


def prepare_image(image, undo_normalization):
    """Convert a normalized tensor into a displayable image."""

    image = undo_normalization(image)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    return np.clip(image, 0, 1)


def save_explanation_figure(
    input_image,
    memory_images,
    memory_labels,
    memory_weights,
    undo_normalization,
    absolute_index,
    true_class,
    original_prediction,
    new_prediction,
    condition,
    output_path,
):
    """Save the input and its ten most influential memory samples."""

    weights = memory_weights[0]

    sorted_positions = torch.argsort(
        weights,
        descending=True,
    ).tolist()

    top_positive_positions = []

    for position in sorted_positions:
        if float(weights[position]) <= 0:
            break

        top_positive_positions.append(position)

        if len(top_positive_positions) >= 10:
            break

    memory_class_counts = torch.bincount(
        memory_labels,
        minlength=10,
    ).tolist()

    fig, axes = plt.subplots(
        2,
        6,
        figsize=(15, 7),
        dpi=180,
    )

    for axis in axes.flat:
        axis.axis("off")

    axes[0, 0].imshow(
        prepare_image(
            input_image,
            undo_normalization,
        )
    )
    axes[0, 0].set_title("Test input")

    count_text = "Memory counts\n" + "\n".join(
        "{}: {}".format(class_number, count)
        for class_number, count in enumerate(
            memory_class_counts
        )
    )

    axes[1, 0].text(
        0.05,
        0.95,
        count_text,
        verticalalignment="top",
        fontsize=9,
    )

    memory_axes = [
        axes[0, 1],
        axes[0, 2],
        axes[0, 3],
        axes[0, 4],
        axes[0, 5],
        axes[1, 1],
        axes[1, 2],
        axes[1, 3],
        axes[1, 4],
        axes[1, 5],
    ]

    for rank, memory_position in enumerate(
        top_positive_positions
    ):
        axis = memory_axes[rank]

        axis.imshow(
            prepare_image(
                memory_images[memory_position],
                undo_normalization,
            )
        )

        axis.set_title(
            "Rank {}\nClass {} | Weight {:.3f}".format(
                rank + 1,
                int(memory_labels[memory_position]),
                float(weights[memory_position]),
            ),
            fontsize=9,
        )

    corrected = new_prediction == true_class

    fig.suptitle(
        "Index {} | {}\n"
        "True: {} | Task 6 prediction: {} | "
        "New prediction: {} | Corrected: {}".format(
            absolute_index,
            condition,
            true_class,
            original_prediction,
            new_prediction,
            corrected,
        ),
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output_path)
    plt.close(fig)

    print("Saved: {}".format(output_path))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Task 7 comparison figures"
    )

    parser.add_argument("--path_model", required=True)
    parser.add_argument("--dir_dataset", required=True)
    parser.add_argument("--targets_csv", required=True)
    parser.add_argument("--random_results_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--examples",
        default="43,68,70",
        help="Comma-separated absolute indices",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print("Device: {}".format(device))

    os.makedirs(args.output_dir, exist_ok=True)

    model, checkpoint = load_model(
        args.path_model,
        device,
    )

    targets = {
        target["absolute_index"]: target
        for target in load_targets(args.targets_csv)
    }

    random_results = load_results(
        args.random_results_csv
    )

    example_indices = [
        int(value.strip())
        for value in args.examples.split(",")
    ]

    train_dataset, _, test_dataset = (
        datasets.get_SVHN_dataset(
            args.dir_dataset,
            size_train=checkpoint["train_examples"],
            seed=42,
        )
    )

    class_indices = group_training_indices_by_class(
        train_dataset,
        checkpoint["num_classes"],
    )

    undo_normalization = datasets.undo_normalization_SVHN

    for absolute_index in example_indices:
        target = targets[absolute_index]
        true_class = target["true_class"]
        original_prediction = target[
            "task6_predicted_class"
        ]

        input_image, dataset_label = test_dataset[
            absolute_index
        ]
        dataset_label = int(dataset_label)

        if dataset_label != true_class:
            raise ValueError(
                "Unexpected label for index {}".format(
                    absolute_index
                )
            )

        # Correct-class memory
        correct_class_seed = args.seed + true_class

        correct_memory, correct_labels, _ = (
            build_correct_class_memory(
                train_dataset,
                class_indices,
                true_class,
                args.memory_size,
                correct_class_seed,
            )
        )

        (
            correct_prediction,
            correct_weights,
            _,
        ) = predict_with_memory(
            model,
            input_image,
            correct_memory,
            device,
        )

        correct_output = os.path.join(
            args.output_dir,
            "index_{}_correct_class_memory.png".format(
                absolute_index
            ),
        )

        save_explanation_figure(
            input_image=input_image,
            memory_images=correct_memory,
            memory_labels=correct_labels,
            memory_weights=correct_weights,
            undo_normalization=undo_normalization,
            absolute_index=absolute_index,
            true_class=true_class,
            original_prediction=original_prediction,
            new_prediction=correct_prediction,
            condition="Correct-class memory",
            output_path=correct_output,
        )

        # Reproduce the successful random set when available.
        random_row = random_results[absolute_index]
        random_corrected = (
            int(random_row["random_corrected"]) == 1
        )

        if random_corrected:
            random_seed = int(
                random_row["successful_seed"]
            )
            random_condition = "Successful random memory"
        else:
            # Reproduce the first unsuccessful random trial.
            random_seed = (
                args.seed
                + absolute_index * 100000
                + 1
            )
            random_condition = (
                "Random memory trial 1 (not corrected)"
            )

        random_memory, random_labels, _ = (
            build_random_memory(
                train_dataset,
                args.memory_size,
                random_seed,
            )
        )

        (
            random_prediction,
            random_weights,
            _,
        ) = predict_with_memory(
            model,
            input_image,
            random_memory,
            device,
        )

        random_output = os.path.join(
            args.output_dir,
            "index_{}_random_memory.png".format(
                absolute_index
            ),
        )

        save_explanation_figure(
            input_image=input_image,
            memory_images=random_memory,
            memory_labels=random_labels,
            memory_weights=random_weights,
            undo_normalization=undo_normalization,
            absolute_index=absolute_index,
            true_class=true_class,
            original_prediction=original_prediction,
            new_prediction=random_prediction,
            condition=random_condition,
            output_path=random_output,
        )

    print()
    print(
        "Finished generating figures in {}".format(
            args.output_dir
        )
    )


if __name__ == "__main__":
    main()