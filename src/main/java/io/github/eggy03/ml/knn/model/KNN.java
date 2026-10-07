package io.github.eggy03.ml.knn.model;

import io.github.eggy03.ml.knn.entity.FeatureVector;
import io.github.eggy03.ml.knn.entity.Neighbor;
import io.github.eggy03.ml.knn.entity.Sample;
import io.github.eggy03.ml.knn.util.DistanceUtils;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Unmodifiable;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

public final class KNN<L> {

    private final int k;
    private final @Unmodifiable @NotNull List<@NotNull Sample<L>> trainingData;

    /**
     * Constructs a KNN model with the training data it needs to memorize
     *
     * @param k            the number of nearest neighbors to consult while predicting
     * @param trainingData the training data based on which predictions will happen
     */
    public KNN(int k, @NotNull List<@NotNull Sample<L>> trainingData) {
        this.k = k;
        this.trainingData = Objects
                .requireNonNull(trainingData, "trainingData cannot be null")
                .stream()
                .map(sample -> Objects.requireNonNull(sample, "sample cannot be null"))
                .toList();

        if (k < 1)
            throw new IllegalArgumentException("Neighbors to consider cannot be less than 1");

        if (k > trainingData.size())
            throw new IllegalArgumentException("Neighbors to consider cannot be greater than the total number of samples provided");
    }

    public @NotNull L predict(@NotNull FeatureVector queryVector) {
        Objects.requireNonNull(queryVector, "queryVector cannot be null");

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
}
