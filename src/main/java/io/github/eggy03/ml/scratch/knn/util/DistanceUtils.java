package io.github.eggy03.ml.scratch.knn.util;


import io.github.eggy03.ml.scratch.knn.entity.FeatureVector;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;

@UtilityClass
@Slf4j
public final class DistanceUtils {

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

        double euclideanDistance = Math.sqrt(sum);
        log.debug("Euclidean Distance between {} and {} is {}", featureVectorOneFeatures, featureVectorTwoFeatures, euclideanDistance);
        return euclideanDistance;

    }
}
