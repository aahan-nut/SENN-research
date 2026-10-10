import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)

if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

from analyze_memory_impact import (
    load_model,
    load_targets,
    predict_with_memory,
)

import utils.datasets as datasets


def encode_training_dataset(
    model,
    train_dataset,
    device,
    batch_size,
):
    """Encode every training sample without using its label."""

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    embedding_batches = []
    label_batches = []

    with torch.no_grad():
        for batch_number, (images, labels) in enumerate(
            loader,
            start=1,
        ):
            images = images.to(device)

            embeddings = model.forward_encoder(images)
            embeddings = embeddings.cpu()

            embedding_batches.append(embeddings)
            label_batches.append(labels.cpu())

            print(
                "Encoding training batch {}/{}".format(
                    batch_number,
                    len(loader),
                ),
                end="\r",
            )

    print()

    return (
        torch.cat(embedding_batches, dim=0),
        torch.cat(label_batches, dim=0),
    )


def load_or_create_embedding_cache(
    model,
    train_dataset,
    device,
    batch_size,
    cache_path,
):
    """Load cached embeddings or create them once."""

    if os.path.exists(cache_path):
        print("Loading embedding cache: {}".format(cache_path))

        cache = torch.load(
            cache_path,
            map_location="cpu",
        )

        return cache["embeddings"], cache["labels"]

    print("Creating training embedding cache...")

    embeddings, labels = encode_training_dataset(
        model,
        train_dataset,
        device,
        batch_size,
    )

    cache_directory = os.path.dirname(
        os.path.abspath(cache_path)
    )
    os.makedirs(cache_directory, exist_ok=True)

    torch.save(
        {
            "embeddings": embeddings,
            "labels": labels,
        },
        cache_path,
    )

    print("Saved embedding cache: {}".format(cache_path))

    return embeddings, labels


def encode_input(model, input_image, device):
    """Encode one test input."""

    input_batch = input_image.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.forward_encoder(input_batch)

    return embedding.cpu()


def calculate_similarities(
    input_embedding,
    training_embeddings,
):
    """Calculate cosine similarity to every training sample."""

    normalized_input = functional.normalize(
        input_embedding,
        p=2,
        dim=1,
    )

    normalized_training = functional.normalize(
        training_embeddings,
        p=2,
        dim=1,
    )

    similarities = torch.matmul(
        normalized_training,
        normalized_input.transpose(0, 1),
    ).squeeze(1)

    return similarities


def select_global_nearest(
    similarities,
    memory_size,
):
    """Select the globally most similar training samples."""

    return torch.topk(
        similarities,
        k=memory_size,
        largest=True,
    ).indices.tolist()


def select_balanced_nearest(
    similarities,
    training_labels,
    samples_per_class,
    num_classes,
):
    """Select equally many nearest samples from each memory class."""

    selected_indices = []

    for class_number in range(num_classes):
        class_positions = torch.where(
            training_labels == class_number
        )[0]

        class_similarities = similarities[
            class_positions
        ]

        nearest_within_class = torch.topk(
            class_similarities,
            k=samples_per_class,
            largest=True,
        ).indices

        selected_indices.extend(
            class_positions[
                nearest_within_class
            ].tolist()
        )

    return selected_indices


def select_excluding_prediction(
    similarities,
    training_labels,
    excluded_prediction,
    memory_size,
):
    """Select nearest samples excluding the original predicted class."""

    allowed_positions = torch.where(
        training_labels != excluded_prediction
    )[0]

    allowed_similarities = similarities[
        allowed_positions
    ]

    nearest_allowed = torch.topk(
        allowed_similarities,
        k=memory_size,
        largest=True,
    ).indices

    return allowed_positions[
        nearest_allowed
    ].tolist()


def build_memory_from_indices(
    train_dataset,
    selected_indices,
):
    """Load memory images and labels from selected dataset positions."""

    memory_images = []
    memory_labels = []

    for dataset_index in selected_indices:
        image, label = train_dataset[dataset_index]

        memory_images.append(image)
        memory_labels.append(int(label))

    return (
        torch.stack(memory_images),
        torch.tensor(memory_labels),
    )


def get_top_positive_labels(
    memory_weights,
    memory_labels,
    limit=10,
):
    """Return labels of the most positively weighted memories."""

    weights = memory_weights[0]

    sorted_positions = torch.argsort(
        weights,
        descending=True,
    ).tolist()

    labels = []

    for position in sorted_positions:
        if float(weights[position]) <= 0:
            break

        labels.append(int(memory_labels[position]))

        if len(labels) >= limit:
            break

    return labels


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Task 8 label-safe memory selection experiment"
    )

    parser.add_argument("--path_model", required=True)
    parser.add_argument("--dir_dataset", required=True)
    parser.add_argument("--targets_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--embedding_cache", required=True)

    parser.add_argument(
        "--memory_size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--target_limit",
        type=int,
        default=1,
        help="Use 0 for all targets",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Device: {}".format(device))

    model, checkpoint = load_model(
        args.path_model,
        device,
    )

    if checkpoint["dataset_name"] != "SVHN":
        raise ValueError("Task 8 currently expects SVHN")

    targets = load_targets(args.targets_csv)

    train_dataset, _, test_dataset = (
        datasets.get_SVHN_dataset(
            args.dir_dataset,
            size_train=checkpoint["train_examples"],
            seed=42,
        )
    )

    training_embeddings, training_labels = (
        load_or_create_embedding_cache(
            model,
            train_dataset,
            device,
            args.embedding_batch_size,
            args.embedding_cache,
        )
    )

    if args.target_limit > 0:
        selected_targets = targets[:args.target_limit]
    else:
        selected_targets = targets

    results = []

    for target_number, target in enumerate(
        selected_targets,
        start=1,
    ):
        absolute_index = target["absolute_index"]
        original_prediction = target[
            "task6_predicted_class"
        ]

        # Retrieve the image but deliberately ignore its true label
        # while selecting all three memory sets.
        input_image, _ = test_dataset[absolute_index]

        input_embedding = encode_input(
            model,
            input_image,
            device,
        )

        similarities = calculate_similarities(
            input_embedding,
            training_embeddings,
        )

        # None of these selection functions receives the true label.
        strategy_indices = {
            "global_nearest": select_global_nearest(
                similarities,
                args.memory_size,
            ),
            "balanced_nearest": select_balanced_nearest(
                similarities,
                training_labels,
                samples_per_class=(
                    args.memory_size
                    // checkpoint["num_classes"]
                ),
                num_classes=checkpoint["num_classes"],
            ),
            "exclude_original_prediction": (
                select_excluding_prediction(
                    similarities,
                    training_labels,
                    original_prediction,
                    args.memory_size,
                )
            ),
        }

        # Access the true label only after memory selection.
        _, true_class = test_dataset[absolute_index]
        true_class = int(true_class)

        if true_class != target["true_class"]:
            raise ValueError(
                "Unexpected true class for index {}".format(
                    absolute_index
                )
            )

        print()
        print(
            "[{}/{}] Index {} | Original prediction {} | True {}".format(
                target_number,
                len(selected_targets),
                absolute_index,
                original_prediction,
                true_class,
            )
        )

        for strategy_name, selected_indices in (
            strategy_indices.items()
        ):
            memory_images, memory_labels = (
                build_memory_from_indices(
                    train_dataset,
                    selected_indices,
                )
            )

            (
                new_prediction,
                memory_weights,
                positive_count,
            ) = predict_with_memory(
                model,
                input_image,
                memory_images,
                device,
            )

            changed = (
                new_prediction != original_prediction
            )
            corrected = new_prediction == true_class

            class_counts = torch.bincount(
                memory_labels,
                minlength=checkpoint["num_classes"],
            ).tolist()

            top_positive_labels = (
                get_top_positive_labels(
                    memory_weights,
                    memory_labels,
                )
            )

            results.append(
                {
                    "absolute_index": absolute_index,
                    "strategy": strategy_name,
                    "true_class": true_class,
                    "original_prediction": original_prediction,
                    "new_prediction": new_prediction,
                    "prediction_changed": int(changed),
                    "prediction_corrected": int(corrected),
                    "positive_memory_weights": positive_count,
                    "memory_class_counts": ";".join(
                        str(value)
                        for value in class_counts
                    ),
                    "top_positive_labels": ";".join(
                        str(value)
                        for value in top_positive_labels
                    ),
                    "selected_memory_indices": ";".join(
                        str(value)
                        for value in selected_indices
                    ),
                }
            )

            print(
                "  {} -> prediction {} | changed {} | corrected {}".format(
                    strategy_name,
                    new_prediction,
                    changed,
                    corrected,
                )
            )

    output_directory = os.path.dirname(
        os.path.abspath(args.output_csv)
    )
    os.makedirs(output_directory, exist_ok=True)

    fieldnames = [
        "absolute_index",
        "strategy",
        "true_class",
        "original_prediction",
        "new_prediction",
        "prediction_changed",
        "prediction_corrected",
        "positive_memory_weights",
        "memory_class_counts",
        "top_positive_labels",
        "selected_memory_indices",
    ]

    with open(args.output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Results saved to: {}".format(args.output_csv))


if __name__ == "__main__":
    main()