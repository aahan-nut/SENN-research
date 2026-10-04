import argparse
import csv
import os
import random
import sys

import numpy as np
import torch


# Make the paper directory importable regardless of the current terminal folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)

if PAPER_DIR not in sys.path:
    sys.path.insert(0, PAPER_DIR)

import utils.datasets as datasets
import utils.utils as utils


def load_targets(csv_path):
    """Load the Task 6 target indices and labels from a CSV file."""

    targets = []

    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            targets.append(
                {
                    "absolute_index": int(row["absolute_index"]),
                    "true_class": int(row["true_class"]),
                    "task6_predicted_class": int(
                        row["task6_predicted_class"]
                    ),
                }
            )

    return targets


def load_model(checkpoint_path, device):
    """Load the pretrained Memory Wrap model."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    modality = checkpoint["modality"]

    if modality not in ["memory", "encoder_memory"]:
        raise ValueError(
            "Unsupported model modality: {}".format(modality)
        )

    model = utils.get_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        model_type=modality,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def build_random_memory(
    train_dataset,
    memory_size,
    seed,
):
    """Create one reproducible random memory set."""

    if memory_size > len(train_dataset):
        raise ValueError(
            "Memory size {} is larger than the training dataset".format(
                memory_size
            )
        )

    random_generator = random.Random(seed)

    selected_indices = random_generator.sample(
        range(len(train_dataset)),
        memory_size,
    )

    memory_images = []
    memory_labels = []

    for dataset_index in selected_indices:
        image, label = train_dataset[dataset_index]
        memory_images.append(image)
        memory_labels.append(int(label))

    memory_tensor = torch.stack(memory_images)
    label_tensor = torch.tensor(memory_labels)

    return memory_tensor, label_tensor, selected_indices

def group_training_indices_by_class(
    train_dataset,
    num_classes,
):
    """Record which training-dataset positions belong to each class."""

    class_indices = {
        class_number: []
        for class_number in range(num_classes)
    }

    for dataset_index in range(len(train_dataset)):
        _, label = train_dataset[dataset_index]
        label = int(label)

        class_indices[label].append(dataset_index)

    return class_indices


def build_correct_class_memory(
    train_dataset,
    class_indices,
    target_class,
    memory_size,
    seed,
):
    """Build memory using only samples from the target's true class."""

    available_indices = class_indices[target_class]

    if len(available_indices) < memory_size:
        raise ValueError(
            "Class {} has only {} samples, but memory size is {}".format(
                target_class,
                len(available_indices),
                memory_size,
            )
        )

    random_generator = random.Random(seed)

    selected_indices = random_generator.sample(
        available_indices,
        memory_size,
    )

    memory_images = []
    memory_labels = []

    for dataset_index in selected_indices:
        image, label = train_dataset[dataset_index]
        memory_images.append(image)
        memory_labels.append(int(label))

    memory_tensor = torch.stack(memory_images)
    label_tensor = torch.tensor(memory_labels)

    return memory_tensor, label_tensor, selected_indices

def predict_with_memory(
    model,
    input_image,
    memory_images,
    device,
):
    """Test one input using one provided memory set."""

    input_batch = input_image.unsqueeze(0).to(device)
    memory_batch = memory_images.to(device)

    with torch.no_grad():
        outputs, memory_weights = model(
            input_batch,
            memory_batch,
            return_weights=True,
        )

    predicted_class = int(
        torch.argmax(outputs, dim=1).item()
    )

    positive_weight_count = int(
        (memory_weights[0] > 0).sum().item()
    )

    return predicted_class, memory_weights.cpu(), positive_weight_count

def find_correcting_random_memory(
    model,
    input_image,
    true_class,
    train_dataset,
    memory_size,
    max_trials,
    base_seed,
    device,
    absolute_index,
):
    """Search random memory sets until one produces the true class."""

    last_prediction = None

    for trial_number in range(1, max_trials + 1):
        memory_seed = base_seed + trial_number

        memory_images, memory_labels, memory_indices = (
            build_random_memory(
                train_dataset,
                memory_size,
                memory_seed,
            )
        )

        prediction, memory_weights, positive_count = (
            predict_with_memory(
                model,
                input_image,
                memory_images,
                device,
            )
        )

        last_prediction = prediction

        if trial_number == 1 or trial_number % 25 == 0:
            print(
                "Index {}: random trial {}/{}".format(
                    absolute_index,
                    trial_number,
                    max_trials,
                ),
                end="\r",
            )

        if prediction == true_class:
            class_counts = torch.bincount(
                memory_labels,
                minlength=10,
            ).tolist()

            weights = memory_weights[0]
            sorted_memory_positions = torch.argsort(
                weights,
                descending=True,
            ).tolist()

            top_positive_labels = []

            for memory_position in sorted_memory_positions:
                weight = float(weights[memory_position])

                if weight <= 0:
                    break

                top_positive_labels.append(
                    int(memory_labels[memory_position])
                )

                if len(top_positive_labels) >= 10:
                    break

            return {
                "corrected": True,
                "trials_tested": trial_number,
                "successful_seed": memory_seed,
                "prediction": prediction,
                "positive_memory_weights": positive_count,
                "memory_class_counts": class_counts,
                "top_positive_labels": top_positive_labels,
                "memory_indices": memory_indices,
            }

    return {
        "corrected": False,
        "trials_tested": max_trials,
        "successful_seed": "",
        "prediction": last_prediction,
        "positive_memory_weights": "",
        "memory_class_counts": [""] * 10,
        "top_positive_labels": [],
        "memory_indices": [],
    }
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Test selected SVHN inputs with different memory sets"
        )
    )

    parser.add_argument(
        "--path_model",
        required=True,
        help="Path to the pretrained model checkpoint",
    )
    parser.add_argument(
        "--dir_dataset",
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--targets_csv",
        required=True,
        help="Path to the Task 7 target CSV file",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=100,
        help="Number of training samples in each memory set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting random seed",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path where correct-class results will be saved",
    )
    parser.add_argument(
        "--random_output_csv",
        required=True,
        help="Path where random-memory results will be saved",
    )
    parser.add_argument(
        "--max_random_trials",
        type=int,
        default=100,
        help="Maximum random memory sets tested for each input",
    )
    parser.add_argument(
        "--target_limit",
        type=int,
        default=0,
        help=(
            "Number of targets to test; use 0 to test all targets"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Fix seeds to make the experiment reproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Device: {}".format(device))

    targets = load_targets(args.targets_csv)

    if len(targets) != 20:
        raise ValueError(
            "Expected 20 Task 7 targets, found {}".format(
                len(targets)
            )
        )

    print("Loaded {} Task 7 targets".format(len(targets)))

    model, checkpoint = load_model(
        args.path_model,
        device,
    )

    print("Dataset: {}".format(checkpoint["dataset_name"]))
    print("Model: {}".format(checkpoint["model_name"]))
    print("Modality: {}".format(checkpoint["modality"]))

    if checkpoint["dataset_name"] != "SVHN":
        raise ValueError(
            "This checkpoint test currently expects SVHN"
        )

    train_dataset, _, test_dataset = datasets.get_SVHN_dataset(
        args.dir_dataset,
        size_train=checkpoint["train_examples"],
        seed=42,
    )

    # For Checkpoint 2, test only the first selected Task 6 input.
    print()
    print("Grouping training examples by class...")

    class_indices = group_training_indices_by_class(
        train_dataset,
        checkpoint["num_classes"],
    )

    for class_number in range(checkpoint["num_classes"]):
        print(
            "Class {}: {} available training samples".format(
                class_number,
                len(class_indices[class_number]),
            )
        )

    results = []
    corrected_count = 0

    print()
    print("Testing correct-class memory sets")
    print("---------------------------------")

    for target_number, target in enumerate(targets, start=1):
        absolute_index = target["absolute_index"]
        expected_true_class = target["true_class"]

        input_image, dataset_true_class = test_dataset[
            absolute_index
        ]
        dataset_true_class = int(dataset_true_class)

        if dataset_true_class != expected_true_class:
            raise ValueError(
                "Index {}: CSV class {} does not match dataset class {}".format(
                    absolute_index,
                    expected_true_class,
                    dataset_true_class,
                )
            )

        # Use one reproducible correct-class memory set.
        # Adding the class number gives each class a separate seed.
        memory_seed = args.seed + dataset_true_class

        memory_images, memory_labels, memory_indices = (
            build_correct_class_memory(
                train_dataset,
                class_indices,
                dataset_true_class,
                args.memory_size,
                memory_seed,
            )
        )

        # Confirm that every memory sample has the true class.
        if not bool(
            torch.all(memory_labels == dataset_true_class)
        ):
            raise ValueError(
                "Correct-class memory contains an incorrect label"
            )

        prediction, memory_weights, positive_count = (
            predict_with_memory(
                model,
                input_image,
                memory_images,
                device,
            )
        )

        corrected = prediction == dataset_true_class

        if corrected:
            corrected_count += 1

        results.append(
            {
                "absolute_index": absolute_index,
                "true_class": dataset_true_class,
                "task6_predicted_class": target[
                    "task6_predicted_class"
                ],
                "correct_class_memory_prediction": prediction,
                "corrected": int(corrected),
                "positive_memory_weights": positive_count,
                "memory_size": args.memory_size,
                "memory_seed": memory_seed,
            }
        )

        print(
            "[{}/{}] Index {} | True {} | Task 6 prediction {} "
            "| Correct-class prediction {} | Corrected: {}".format(
                target_number,
                len(targets),
                absolute_index,
                dataset_true_class,
                target["task6_predicted_class"],
                prediction,
                corrected,
            )
        )

    output_directory = os.path.dirname(
        os.path.abspath(args.output_csv)
    )
    os.makedirs(output_directory, exist_ok=True)

    fieldnames = [
        "absolute_index",
        "true_class",
        "task6_predicted_class",
        "correct_class_memory_prediction",
        "corrected",
        "positive_memory_weights",
        "memory_size",
        "memory_seed",
    ]

    with open(args.output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("Correct-class memory experiment complete")
    print("----------------------------------------")
    print(
        "Corrected predictions: {}/{}".format(
            corrected_count,
            len(targets),
        )
    )
    print(
        "Results saved to: {}".format(
            args.output_csv
        )
    )
    print()
    print("Starting random-memory search")
    print("-----------------------------")

    if args.target_limit > 0:
        random_targets = targets[:args.target_limit]
    else:
        random_targets = targets

    random_results = []
    random_corrected_count = 0

    for target_number, target in enumerate(
        random_targets,
        start=1,
    ):
        absolute_index = target["absolute_index"]
        true_class = target["true_class"]

        input_image, dataset_true_class = test_dataset[
            absolute_index
        ]
        dataset_true_class = int(dataset_true_class)

        if dataset_true_class != true_class:
            raise ValueError(
                "Index {} has unexpected true class {}".format(
                    absolute_index,
                    dataset_true_class,
                )
            )

        # Give every target a separate, reproducible seed range.
        base_seed = args.seed + (absolute_index * 100000)

        print()
        print(
            "[{}/{}] Searching random memories for index {} "
            "(true class {})".format(
                target_number,
                len(random_targets),
                absolute_index,
                true_class,
            )
        )

        search_result = find_correcting_random_memory(
            model=model,
            input_image=input_image,
            true_class=true_class,
            train_dataset=train_dataset,
            memory_size=args.memory_size,
            max_trials=args.max_random_trials,
            base_seed=base_seed,
            device=device,
            absolute_index=absolute_index,
        )

        if search_result["corrected"]:
            random_corrected_count += 1
            status = "CORRECTED"
        else:
            status = "NOT CORRECTED"

        print()
        print(
            "Index {}: {} after {} trial(s)".format(
                absolute_index,
                status,
                search_result["trials_tested"],
            )
        )

        class_counts = search_result["memory_class_counts"]

        result_row = {
            "absolute_index": absolute_index,
            "true_class": true_class,
            "task6_predicted_class": target[
                "task6_predicted_class"
            ],
            "random_corrected": int(
                search_result["corrected"]
            ),
            "trials_tested": search_result["trials_tested"],
            "successful_seed": search_result[
                "successful_seed"
            ],
            "successful_prediction": search_result[
                "prediction"
            ],
            "positive_memory_weights": search_result[
                "positive_memory_weights"
            ],
            "top_positive_labels": ";".join(
                str(label)
                for label in search_result[
                    "top_positive_labels"
                ]
            ),
            "memory_indices": ";".join(
                str(index)
                for index in search_result["memory_indices"]
            ),
        }

        for class_number in range(10):
            result_row[
                "memory_class_{}_count".format(class_number)
            ] = class_counts[class_number]

        random_results.append(result_row)

    random_output_directory = os.path.dirname(
        os.path.abspath(args.random_output_csv)
    )
    os.makedirs(random_output_directory, exist_ok=True)

    random_fieldnames = [
        "absolute_index",
        "true_class",
        "task6_predicted_class",
        "random_corrected",
        "trials_tested",
        "successful_seed",
        "successful_prediction",
        "positive_memory_weights",
        "top_positive_labels",
        "memory_indices",
    ]

    for class_number in range(10):
        random_fieldnames.append(
            "memory_class_{}_count".format(class_number)
        )

    with open(
        args.random_output_csv,
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=random_fieldnames,
        )
        writer.writeheader()
        writer.writerows(random_results)

    print()
    print("Random-memory search complete")
    print("-----------------------------")
    print(
        "Corrected predictions: {}/{}".format(
            random_corrected_count,
            len(random_targets),
        )
    )
    print(
        "Results saved to: {}".format(
            args.random_output_csv
        )
    )

if __name__ == "__main__":
    main()