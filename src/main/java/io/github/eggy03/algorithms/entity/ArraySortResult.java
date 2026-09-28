package io.github.eggy03.algorithms.entity;

import org.jetbrains.annotations.NotNull;

import java.math.BigDecimal;
import java.util.List;
import java.util.Objects;

/**
 * Represents a sorted (unmodifiable) array and the time taken to sort the given array (in ms)
 * @param sortedArray
 * @param sortTime
 */
public record ArraySortResult(List<?> sortedArray, BigDecimal sortTime) {

    public ArraySortResult(@NotNull List<?> sortedArray, @NotNull BigDecimal sortTime) {
        this.sortedArray = List.copyOf(sortedArray);
        this.sortTime = Objects.requireNonNull(sortTime, "sortTime cannot be null");
    }

    @Override
    public @NotNull String toString() {
        return "ArraySortResult{" +
                "problemSize=" + sortedArray.size() +
                " sortedArray=" + sortedArray +
                ", sortTime=" + sortTime + " ms" +
                '}';
    }
}
