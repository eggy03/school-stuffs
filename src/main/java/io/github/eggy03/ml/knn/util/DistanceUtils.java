package io.github.eggy03.ml.knn.util;


import io.github.eggy03.ml.knn.entity.FeatureVector;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;

public final class DistanceUtils {

    private DistanceUtils() {
        throw new IllegalStateException("Utility Class");
    }

    public static double euclideanDistance(@NotNull FeatureVector featureVectorOne, @NotNull FeatureVector featureVectorTwo) {

        Objects.requireNonNull(featureVectorOne, "featureVectorOne cannot be null");
        Objects.requireNonNull(featureVectorTwo, "featureVectorTwo cannot be null");

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
