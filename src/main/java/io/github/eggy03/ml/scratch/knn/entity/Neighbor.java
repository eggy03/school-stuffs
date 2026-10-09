package io.github.eggy03.ml.scratch.knn.entity;

import org.jetbrains.annotations.NotNull;

import java.util.Objects;

/**
 * A {@code Neighbor} is a combination of a {@link Sample}
 * and the {@code distance} of it's {@link FeatureVector}
 * from another sample's feature vector
 *
 * @param <L> the type of the label associated with the sample
 * @param distance the calculated distance between the query feature vector
 *                 and this neighbor's feature vector
 * @param sample the neighboring sample; must not be {@code null}
 *
 */
public record Neighbor<L>(double distance, @NotNull Sample<L> sample) {

    public Neighbor(double distance, @NotNull Sample<L> sample) {
        this.distance = distance;
        this.sample = Objects.requireNonNull(sample, "sample cannot be null");
    }
}
