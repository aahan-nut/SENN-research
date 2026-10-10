# Task 8 Experiment Plan

## Constraint

The true label of a selected test input must never be used to select
its memory set. The true label will be accessed only after memory
selection to evaluate whether the resulting prediction is correct.

All strategies will be applied consistently to all 20 selected inputs.

## Strategy 1: Global nearest neighbors

Encode the test input and all 2,000 training samples using the
pretrained model's encoder. Use cosine similarity to select the 100
training samples most similar to the input.

This strategy uses no labels during memory selection.

## Strategy 2: Balanced nearest neighbors

For each memory-sample class from 0 through 9, select the 10 samples
whose encoder embeddings are most similar to the input. Combine these
samples into a balanced memory set of 100 examples.

This uses labels associated with the training-memory samples, which is
allowed, but it does not use the test input's true label.

## Strategy 3: Exclude the original predicted class

Use the incorrect Task 6 prediction as an observable model output.
Exclude training-memory samples carrying that predicted class, then
select the 100 remaining samples most similar to the input.

This strategy does not use the test input's true label. It tests
whether removing support for the model's original prediction encourages
a different and potentially correct prediction.

## Evaluation

For each strategy and each of the 20 inputs, record:

- Absolute dataset index
- Original Task 6 prediction
- New prediction
- Whether the prediction changed
- Whether the new prediction is correct
- Memory-set class composition
- Labels of the most positively weighted memory samples

The true class will be compared with the new prediction only after the
memory set has already been selected.