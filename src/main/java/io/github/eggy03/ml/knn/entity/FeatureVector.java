package io.github.eggy03.ml.knn.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Unmodifiable;

import java.util.List;
import java.util.Objects;

/**
 * A list of numeric featureList for a given {@link Sample}
 * @param featureList a list of numeric featureList associated with the sample
 */
public record FeatureVector(@NotNull List<@NotNull Number> featureList) {

    public FeatureVector(@NotNull List<@NotNull Number> featureList) {

        this.featureList = Objects.requireNonNull(featureList, "feature list cannot be null")
                .stream()
                .map(feature -> Objects.requireNonNull(feature, "feature cannot be null"))
                .toList();
    }

    @JsonIgnore
    public boolean isEmpty() {
        return featureList.isEmpty();
    }

    @JsonIgnore
    public int featureSize() {
        return featureList.size();
    }

    @Override
    @Contract(pure = true)
    @JsonIgnore
    public @NotNull @Unmodifiable List<@NotNull Number> featureList() {
        return List.copyOf(featureList);
    }
}
