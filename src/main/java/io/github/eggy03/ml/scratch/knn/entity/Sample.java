package io.github.eggy03.ml.scratch.knn.entity;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.jetbrains.annotations.NotNull;

import java.util.List;
import java.util.Objects;

/**
 * A sample is a unit form of your training data.
 *
 * <p>
 * For example,If a dataset contains {@code Weight} and {@code Height} as featureList
 * and {@code T-Shirt Label} as the label, each row in the dataset represents one {@code Sample}.
 * </p>
 *
 * @param featureVector the {@link FeatureVector} that describes this sample
 * @param label         the label associated with this sample
 * @param <L>           the type of the label
 */
public record Sample<L>(@NotNull FeatureVector featureVector, @NotNull L label) {

    public Sample(@NotNull FeatureVector featureVector, @NotNull L label) {
        this.featureVector = Objects.requireNonNull(featureVector, "featureVector cannot be null");
        this.label = Objects.requireNonNull(label, "label cannot be null");
    }

    @JsonCreator(mode = JsonCreator.Mode.PROPERTIES)
    public Sample(
            @JsonProperty("features") @NotNull List<Number> featureList,
            @JsonProperty("label") @NotNull L label
    ) {
        this(new FeatureVector(featureList), label);
    }
}
