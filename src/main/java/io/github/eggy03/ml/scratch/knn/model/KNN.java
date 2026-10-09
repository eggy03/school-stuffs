package io.github.eggy03.ml.scratch.knn.model;

import io.github.eggy03.ml.scratch.knn.entity.FeatureVector;
import io.github.eggy03.ml.scratch.knn.entity.Neighbor;
import io.github.eggy03.ml.scratch.knn.entity.Sample;
import io.github.eggy03.ml.scratch.knn.util.DistanceUtils;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Unmodifiable;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

@Slf4j
public final class KNN<L> {

    private final int k;
    private final @Unmodifiable
    @NotNull List<@NotNull Sample<L>> trainingData;

    /**
     * Constructs a KNN model with the training data it needs to memorize
     *
     * @param k            the number of nearest neighbors to consult while predicting
     * @param trainingData the training data based on which predictions will happen
     */
    public KNN(int k, @NotNull List<@NotNull Sample<L>> trainingData) {

        if (k < 1)
            throw new IllegalArgumentException("Neighbors to consider cannot be less than 1");

        if (k > trainingData.size())
            throw new IllegalArgumentException("Neighbors to consider cannot be greater than the total number of samples provided");

        this.k = k;
        this.trainingData = Objects
                .requireNonNull(trainingData, "trainingData cannot be null")
                .stream()
                .map(sample -> Objects.requireNonNull(sample, "sample in trainingData cannot be null"))
                .toList();

        log.debug("KNN Model initiated with {} samples, with k={}", trainingData.size(), k);
    }

    public @NotNull L predict(@NotNull FeatureVector queryVector) {
        Objects.requireNonNull(queryVector, "queryVector cannot be null");

        log.debug("Provided Query Vector: {}", queryVector.featureList());

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

        log.debug("For the given query vector, the following {} nearest neighbors have been located:", k);
        nearestNeighborList.forEach(neighbor ->
                log.debug(
                        "Sample -> Features = {} | Label = {} | Distance = {}",
                        neighbor.sample().featureVector().featureList(),
                        neighbor.sample().label(),
                        String.format("%.4f", neighbor.distance())
                )
        );

        // and then create a weight map of the labels
        // which is the sum of the inverse distance of the neighbors
        // grouped by labels
        Map<L, Double> labelWeightMap = nearestNeighborList
                .stream()
                .collect(Collectors.groupingBy(
                                neighbor -> neighbor.sample().label(),
                                Collectors.summingDouble(neighbor -> neighbor.distance() == 0 ? 1.0 / 1e-9 : 1.0 / neighbor.distance())
                        )
                );

        log.debug("The weight-map is as follows:\n{}", labelWeightMap);

        // from this weight map find the label with the largest summed-weight
        return labelWeightMap
                .entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .orElseThrow(() -> new RuntimeException("Could not find the max occurrence of a label"))
                .getKey();
    }
}
