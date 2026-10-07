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

3) Then we will consider `k nearest neighbors/samples` and will take whichever `label` appeared the maximum time.

# Design

## Designing the Entities

### Sample

We will start with designing our `Sample` class. A sample represents a unit row in the dataset.

Keeping that in mind, we create a record of `Sample<L>`, where `L` is the type of the label.

```java

import io.github.eggy03.ml.knn.entity.FeatureVector;

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

Then we design our prediction function which will take in the feature vector and output a label based on the trained data

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
The time complexity of this function is `O(nxm)`, where `n = number of training samples` and `m = number of features in a single sample`.

# Architectural Limitations

1) Features must be numeric
2) All samples must have the same number of features

# Training, Testing and Observation

## Training

We have 150 samples (50 samples per Iris Class), which we will split into `training_dataset` and `testing_dataset`

- For training, we will use 120 of the samples (40 samples from each Iris Class)
- For testing, we will use the remaining 30 (10 samples from each Iris Class)

## Testing

We will test against different values of `k` such as (`3`, `5`, `7`, `9`) and
observe the model's correctness.

Our `testing dataset` is as follows:



## Observation

