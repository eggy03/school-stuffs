# Introduction

This document attempts at capturing my design philosophy and decisions during designing a simple KNN model

For our sample data, we will take a look at the [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

The following table represents a few samples taken from the above dataset:

| Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Label           |
|------------------:|-----------------:|------------------:|-----------------:|-----------------|
|               5.1 |              3.5 |               1.4 |              0.2 | Iris-setosa     |
|               4.9 |              3.0 |               1.4 |              0.2 | Iris-setosa     |
|               4.7 |              3.2 |               1.3 |              0.2 | Iris-setosa     |
|               4.6 |              3.1 |               1.5 |              0.2 | Iris-setosa     |
|               7.0 |              3.2 |               4.7 |              1.4 | Iris-versicolor |
|               6.4 |              3.2 |               4.5 |              1.5 | Iris-versicolor |
|               6.9 |              3.1 |               4.9 |              1.5 | Iris-versicolor |
|               5.5 |              2.3 |               4.0 |              1.3 | Iris-versicolor |
|               6.3 |              3.3 |               6.0 |              2.5 | Iris-virginica  |
|               5.8 |              2.7 |               5.1 |              1.9 | Iris-virginica  |
|               7.1 |              3.0 |               5.9 |              2.1 | Iris-virginica  |
|               6.3 |              2.9 |               5.6 |              1.8 | Iris-virginica  |

Each row in the above table represents a Sample.

A Sample consists of

1) Features - `Sepal Length`, `Sepal Width`, `Petal Length` and `Petal Width`
2) Label - e.g., `Iris-setosa`

# Goal

Our goal is to design our KNN model and make the model study the `samples` in the dataset so that it can predict
the `label` of an unlabelled sample, based on it's `features`.

# Prediction Strategy

Our prediction strategy is simple.

1) We will calculate the `Euclidean Distance` or `Normalized Euclidean Distance`
   between each `dataset sample` and the `query sample`.

2) Then we will sort the `dataset sample` in ascending order of their
   distances from the `query sample`.

3) Then we will consider `k nearest neighbors/samples` and will take whichever `label` appeared the maximum time (majority voting).

# Design

## Designing the Entities

### Sample

We will start with designing our `Sample` class. A sample represents a unit row in the dataset.

Keeping that in mind, we create a record of `Sample<L>`, where `L` is the type of the label.

```java

import io.github.eggy03.ml.scratch.knn.entity.FeatureVector;

public record Sample<L>(FeatureVector featureVector, L label) {
}
```

### Feature Vector

`FeatureVector` is nothing but the *list of features* of a sample. Realistically, features can be of any type.

But for our example, since, we are using `Euclidean Distance` to calculate nearest neighbors,
we will be using the `Number` type.

```java

public record FeatureVector(List<Number> features) {
}
```

### Neighbor

A `Neighbor` represents a training `Sample` and it's distance from a query `Sample`.
A query sample is the sample whose `label` we need to predict.
So, neighbors basically represent the training samples and their distances from the query sample.

```java

public record Neighbor<L>(double distance, Sample<L> sample) {
}
```

## Euclidean Distance Calculation

Previously, we stated in our [Prediction Strategy](#prediction-strategy) that we will calculate the `Euclidean Distance`
between two samples to determine how near/far they are to each other.

The calculation is actually performed between their `feature vectors`.

Suppose, we have a feature vector: `[a, b, c, d]` and another feature vector: `[w, x, y, z]`.

The `Euclidean Distance` between these two vectors is: `√(a-w)² + (b-x)² + (c-y)² + (d-z)²`

If we were to represent these two vectors in an array, such that
`A[] = {a, b, c, d}` and `B[] = {w, x, y, z}`

then, `distance = √(A[0]-B[0])² + (A[1]-B[1])² + (A[2]-B[2])² + (A[3]-B[3])²`.

One thing to note is that, both the vectors must have the same dimensions,
meaning both the feature lists must have the same size.

Based on this data, we design the following:

```java
public final class DistanceUtils {

    public static double euclideanDistance(FeatureVector featureVectorOne, FeatureVector featureVectorTwo) {

        if (featureVectorOne.isEmpty() || featureVectorTwo.isEmpty())
            throw new IllegalStateException("feature vectors cannot be empty");

        if (featureVectorOne.featureSize() != featureVectorTwo.featureSize())
            throw new IllegalArgumentException("Feature vectors must have the same dimensions");

        double sum = 0.0;

        List<Number> featureVectorOneFeatures = featureVectorOne.featureList();
        List<Number> featureVectorTwoFeatures = featureVectorTwo.featureList();

        for (int i = 0; i < featureVectorOne.featureSize(); i++) {
            double difference = featureVectorOneFeatures.get(i).doubleValue() - featureVectorTwoFeatures.get(i).doubleValue();
            sum = sum + Math.pow(difference, 2);
        }

        return Math.sqrt(sum);
    }
}
```

## Designing the Model

We design our constructor which takes in the nearest neighbors to consider and the training data

```java
public final class KNN<L> {

    private final int k;
    private final List<Sample<L>> trainingData;

    public KNN(int k, List<Sample<L>> trainingData) {

        if (k < 1)
            throw new IllegalArgumentException("Neighbors to consider cannot be less than 1");

        if (k > trainingData.size())
            throw new IllegalArgumentException("Neighbors to consider cannot be greater than the total number of samples provided");

        this.k = k;
        this.trainingData = trainingData;
    }
}
```

Then we design our prediction function which will take in the feature vector and output a label based on the trained
data

```java
public L predict(FeatureVector queryVector) {
    // for each sample in the training data,
    // we calculate the Euclidean distance between the training sample feature vector and the query sample feature vector
    // and then map the distance and the training sample into a new neighbor.
    // the resultant neighbors are sorted based on their increasing order of distances from the query sample
    // and first k neighbors are selected
    List<Neighbor<L>> nearestNeighborList = trainingData
            .stream()
            .map(sample -> {
                double euclideanDistance = DistanceUtils.euclideanDistance(sample.featureVector(), queryVector);
                return new Neighbor<>(euclideanDistance, sample);
            }).sorted(Comparator.comparing(Neighbor::distance))
            .limit(k)
            .toList();

    // collect all the labels from the nearest neighbor list
    List<L> nearestNeighborLabelList = nearestNeighborList
            .stream()
            .map(neighbor -> neighbor.sample().label())
            .toList();

    // and then create a frequency map of the labels (Map<Label, OccurrenceCount>)
    Map<L, Long> labelFrequencyMap = nearestNeighborLabelList
            .stream()
            .collect(Collectors.groupingBy(label -> label, Collectors.counting()));

    // from this frequency map find the label with the largest occurrence
    return labelFrequencyMap
            .entrySet()
            .stream()
            .max(Map.Entry.comparingByValue())
            .orElseThrow(() -> new RuntimeException("Could not find the max occurrence of a label"))
            .getKey();
}
```

# Architectural Limitations

1) Features must be numeric
2) All samples must have the same number of features

# Training, Testing, Observation and Inference

## Training

We have 150 samples (50 samples per Iris Class), which we will split into `training_dataset` and `testing_dataset`

- For training, we will use 120 of the samples (40 samples from each Iris Class)
- For testing, we will use the remaining 30 (10 samples from each Iris Class)

## Testing

We will test against different values of `k` such as (`3`, `5`, `7`, `9`) and
observe the model's correctness.

Our `testing_dataset` is as follows:

| Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) | Label           |
|-------------------|------------------|-------------------|------------------|-----------------|
| 5.0               | 3.5              | 1.3               | 0.3              | Iris-setosa     |
| 4.5               | 2.3              | 1.3               | 0.3              | Iris-setosa     |
| 4.4               | 3.2              | 1.3               | 0.2              | Iris-setosa     |
| 5.0               | 3.5              | 1.6               | 0.6              | Iris-setosa     |
| 5.1               | 3.8              | 1.9               | 0.4              | Iris-setosa     |
| 4.8               | 3.0              | 1.4               | 0.3              | Iris-setosa     |
| 5.1               | 3.8              | 1.6               | 0.2              | Iris-setosa     |
| 4.6               | 3.2              | 1.4               | 0.2              | Iris-setosa     |
| 5.3               | 3.7              | 1.5               | 0.2              | Iris-setosa     |
| 5.0               | 3.3              | 1.4               | 0.2              | Iris-setosa     |
| 5.5               | 2.5              | 4.0               | 1.3              | Iris-versicolor |
| 5.5               | 2.6              | 4.4               | 1.2              | Iris-versicolor |
| 6.1               | 3.0              | 4.6               | 1.4              | Iris-versicolor |
| 5.8               | 2.6              | 4.0               | 1.2              | Iris-versicolor |
| 5.0               | 2.3              | 3.3               | 1.0              | Iris-versicolor |
| 5.6               | 2.7              | 4.2               | 1.3              | Iris-versicolor |
| 5.7               | 3.0              | 4.2               | 1.2              | Iris-versicolor |
| 5.7               | 2.9              | 4.2               | 1.3              | Iris-versicolor |
| 6.2               | 2.9              | 4.3               | 1.3              | Iris-versicolor |
| 5.1               | 2.5              | 3.0               | 1.1              | Iris-versicolor |
| 6.7               | 3.1              | 5.6               | 2.4              | Iris-virginica  |
| 6.9               | 3.1              | 5.1               | 2.3              | Iris-virginica  |
| 5.8               | 2.7              | 5.1               | 1.9              | Iris-virginica  |
| 6.8               | 3.2              | 5.9               | 2.3              | Iris-virginica  |
| 6.7               | 3.3              | 5.7               | 2.5              | Iris-virginica  |
| 6.7               | 3.0              | 5.2               | 2.3              | Iris-virginica  |
| 6.3               | 2.5              | 5.0               | 1.9              | Iris-virginica  |
| 6.5               | 3.0              | 5.2               | 2.0              | Iris-virginica  |
| 6.2               | 3.4              | 5.4               | 2.3              | Iris-virginica  |
| 5.9               | 3.0              | 5.1               | 1.8              | Iris-virginica  |

## Observation

| K      | Correct | Accuracy |
|--------|---------|----------|
| 3–45   | 30/30   | 100.00%  |
| 47–49  | 29/30   | 96.67%   |
| 51     | 28/30   | 93.33%   |
| 53–55  | 27/30   | 90.00%   |
| 57–71  | 26/30   | 86.67%   |
| 73–79  | 27/30   | 90.00%   |
| 81     | 20/30   | 66.67%   |
| 83–87  | 19/30   | 63.33%   |
| 89     | 18/30   | 60.00%   |
| 91–119 | 10/30   | 33.33%   |

Initially, testing the 30 samples from the `testing_dataset`, against the `training_dataset` for increasing values of
`K`
shows us that the accuracy of the predictions decreases gradually when the classifier has to consider a large number of
neighbors.

But we still don't know what the incorrect predictions are, as in, if `Iris-setosa` was expected, what did we get
instead.
To know that, we will take the help of the confusion matrix.

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            |             |                 |                |
| Iris-versicolor        |             |                 |                |
| Iris-virginica         |             |                 |                |

We can see that for `K= 3 to 45`, we have a 100% accuracy.

Let's consider the confusion matrices for `k = 7, 45, 51, 81, 91`

### k = 7

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            | 10          | 0               | 0              |
| Iris-versicolor        | 0           | 10              | 0              |
| Iris-virginica         | 0           | 0               | 10             |

### k = 45

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            | 10          | 0               | 0              |
| Iris-versicolor        | 0           | 10              | 0              |
| Iris-virginica         | 0           | 0               | 10             |

### k = 51

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            | 10          | 0               | 0              |
| Iris-versicolor        | 0           | 10              | 0              |
| Iris-virginica         | 0           | 2               | 8              |

### k = 81

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            | 10          | 0               | 0              |
| Iris-versicolor        | 0           | 10              | 0              |
| Iris-virginica         | 0           | 10              | 0              |

### k = 91

| Actual ↓ / Predicted → | Iris-setosa | Iris-versicolor | Iris-virginica |
|------------------------|-------------|-----------------|----------------|
| Iris-setosa            | 0           | 10              | 0              |
| Iris-versicolor        | 0           | 10              | 0              |
| Iris-virginica         | 0           | 10              | 0              |

## Inference

For (k=7) and (k=45), all 30 test samples were classified correctly.

At (k=51), two `Iris-virginica` samples were incorrectly classified as `Iris-versicolor`, reducing the accuracy to
`93.33%`.

At (k=81), all `Iris-virginica` samples were classified as `Iris-versicolor`, while the `Iris-setosa` and
`Iris-versicolor` samples were classified correctly. The accuracy consequently decreased to `66.67%`.

At (k=91), the model classified all 30 test samples as `Iris-versicolor`, resulting in an accuracy of `33.33%`.

This indicates that the majority-voting mechanism increasingly favored `Iris-versicolor` for the selected test samples when a
large number of neighbors was considered.

### Tie Breaking in Majority Voting

The current model tries to select the label from it's selected neighbors by using `Majority Voting`.

Say, if, for `K=31`, we have:

1) Occurrences of `Iris-setosa`: 19
2) Occurrences of `Iris-versicolor`: 10
3) Occurrences of `Iris-virginica`: 2

`Iris-setosa` will be selected.

But for large values of `K`, such as `K=119`, we may end up with:

1) Occurrences of `Iris-setosa`: 40
2) Occurrences of `Iris-versicolor`: 40
3) Occurrences of `Iris-virginica`: 39

When multiple labels have the same maximum vote count, the current implementation does not define an explicit tie-breaking rule.
Since `Collectors.groupingBy()` uses a `HashMap` by default, the iteration order of the labels is not guaranteed.
Consequently, the selected label in a tie should not be assumed to be deterministic.

To break this tie, we can implement one of the methods mentioned [here](https://thefoundling.net/prof/courses/fall24/daemen/401notes10_22.pdf).

For our model, we will choose `Weighted Voting`, a.k.a., the `Inverse Distance` method.

We re-factor our `predict` function as follows:

```java
public L predict(FeatureVector queryVector) {
        // for each sample in the training data,
        // we calculate the Euclidean distance between the training sample feature vector and the query sample feature vector
        // and then map the distance and the training sample into a new neighbor.
        // the resultant neighbors are sorted based on their increasing order of distances from the query sample
        // and first k neighbors are selected
        List<Neighbor<L>> nearestNeighborList = trainingData
                .stream()
                .map(sample -> {
                    double euclideanDistance = DistanceUtils.euclideanDistance(sample.featureVector(), queryVector);
                    return new Neighbor<>(euclideanDistance, sample);
                }).sorted(Comparator.comparing(Neighbor::distance))
                .limit(k)
                .toList();
        
        // and then create a weight map of the labels
        Map<L, Double> labelWeightMap = nearestNeighborList
                .stream()
                .collect(Collectors.groupingBy(
                                neighbor -> neighbor.sample().label(),
                                Collectors.summingDouble(neighbor -> neighbor.distance() == 0 ? 1.0 / 1e-9 : 1.0 / neighbor.distance())
                        )
                );

        // from this weight map find the label with the largest weight
        return labelWeightMap
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new RuntimeException("Could not find the max occurrence of a label"))
                .getKey();
    }
```
With the new `Weighted Voting` method, we have achieved a 100% accuracy on from `k= 3 to 119`