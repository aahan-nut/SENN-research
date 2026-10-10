# Task 8: Error Correction with Memory-Set Selection

## Objective

The goal of this experiment was to determine whether a consistent
memory-set selection strategy could correct the 20 wrong predictions
identified in Task 6.

The true label of each test image was not used when selecting its
memory set. It was only accessed afterward to evaluate whether the new
prediction was correct. All strategies were applied in the same way to
all 20 test images.

The pretrained Memory Wrap model and a memory-set size of 100 samples
were used for every experiment.

## Strategies

### 1. Global nearest neighbors

I encoded the test input and all 2,000 training samples with the
model's encoder. I calculated cosine similarity and selected the 100
training samples most similar to the test input.

This strategy did not use any labels during memory selection.

### 2. Balanced nearest neighbors

I selected the 10 most similar training samples from each of the 10
classes, producing a balanced memory set of 100 samples.

This strategy used the labels of the training samples, which is
allowed, but it did not use the true label of the test input.

### 3. Exclude the original predicted class

I excluded training samples belonging to the model's original
predicted class. From the remaining samples, I selected the 100 most
similar samples.

This strategy used the model's observable original prediction, but it
did not use the true label of the test input.

## Results

| Strategy | Predictions changed | Predictions corrected |
|---|---:|---:|
| Global nearest neighbors | 2/20 (10%) | 1/20 (5%) |
| Balanced nearest neighbors | 2/20 (10%) | 1/20 (5%) |
| Exclude original prediction | 4/20 (20%) | 1/20 (5%) |

All three strategies corrected the prediction for absolute test index
70. Its true class was 3, while its original prediction was 5. Each
strategy changed its prediction from 5 to 3.

Although excluding the original predicted class changed the largest
number of predictions, it did not correct more predictions. It changed
four predictions but corrected only one. Therefore, changing a
prediction does not necessarily improve its accuracy.

## Example 1: Successful correction at index 70

For index 70, the true class was 3 and the original prediction was 5.

The global-nearest memory set contained 48 samples from class 3 and 44
from class 5. Its most positively weighted samples included both
classes 3 and 5. The resulting prediction changed to class 3.

The balanced strategy also corrected this image even though its memory
set contained exactly 10 samples from every class. This indicates that
the class counts alone do not determine the prediction. The learned
weights assigned by the model to individual memory samples are also
important.

When samples from the original predicted class 5 were excluded, the
selected memory set contained 81 class-3 samples. Its ten most
positively weighted samples were all from class 3, and the prediction
changed to the correct class.

The consistency across all three strategies suggests that this image
was already close to the model's decision boundary between classes 3
and 5. Changing the memory evidence was enough to move it toward the
correct class.

## Example 2: An incorrect change at index 96

For index 96, the true class was 8 and the original prediction was 2.
All three strategies changed the prediction to class 7 rather than the
correct class 8.

The global-nearest memory set contained 61 class-7 samples, and most of
the highest positively weighted samples were also from class 7. The
exclude-original-prediction strategy contained 75 class-7 samples.
This explains why both strategies moved the prediction toward class 7.

The balanced strategy also produced class 7 even though every class
had equal representation. Its most positively weighted samples were
primarily from class 7. This provides further evidence that the
model's learned memory weights matter more than only the raw number of
samples from each class.

## Other prediction changes

The exclude-original-prediction strategy also changed index 32 from
class 6 to class 5 and index 59 from class 0 to class 6. Neither change
was correct.

For index 32, 82 of the 100 selected memory samples belonged to class
5, and most of the strongest positive memory samples were class 5.
This appears to have caused the new class-5 prediction.

For index 59, the selected memory set was more mixed. Class 6 was the
largest group, with 33 samples, and class 5 had 20 samples. The
prediction changed to class 6, but the true class was 5.

## Analysis and conclusions

The experiments show that memory-set selection can influence the
model's output, but the effect is limited and does not reliably improve
accuracy. The most aggressive strategy, excluding the original
predicted class, changed twice as many predictions as the other
strategies but still corrected only one image.

Only index 70 was corrected, and it was corrected by all three
strategies. This suggests that correction depends strongly on the
input and its encoder representation. Inputs near an appropriate
decision boundary may be correctable, while inputs with misleading
representations may simply move from one incorrect class to another.

Based on these 20 examples, I do not believe these strategies provide
a reliable unified method for correcting most predictions. The best
correction rate was only 5%. A more successful strategy may require
better encoder representations, confidence-aware memory selection, or
a learned method for selecting memory samples. However, any future
strategy must continue to avoid using the unknown true label of the
test input.

The results also demonstrate that memory composition alone is not
enough to explain the outcome. The similarity of the selected samples
and the positive weights assigned by the Memory Wrap model both affect
the final prediction.