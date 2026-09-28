package io.github.eggy03.algorithms.sort;

import io.github.eggy03.algorithms.entity.ArraySortResult;
import io.github.eggy03.algorithms.exceptions.AlreadySortedException;
import org.jetbrains.annotations.NotNull;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;

public final class InsertionSort {

    private InsertionSort() {
        throw new IllegalStateException("Utility Class");
    }

    public static @NotNull ArraySortResult ascendingSortArrayOfIntegers(int[] unsortedArray) {

        // create a copy of the array for thread safety
        int[] arrayCopy = Arrays.copyOf(unsortedArray, unsortedArray.length);

        if (arrayCopy.length == 0)
            throw new AlreadySortedException("Array to be sorted cannot be empty");

        if (arrayCopy.length == 1)
            return new ArraySortResult(Arrays.stream(arrayCopy).boxed().toList(), BigDecimal.valueOf(0.00));

        long startTime = System.nanoTime();
        for (int j = 1 ; j < arrayCopy.length ; j++) {
            int key = arrayCopy[j];
            int i = j - 1;

            while (i >= 0 && arrayCopy[i] > key) {
                arrayCopy[i + 1] =  arrayCopy[i];
                i--;
            }

            arrayCopy[i + 1] = key;
        }

        long endTime = System.nanoTime();
        return new ArraySortResult(
                Arrays.stream(arrayCopy).boxed().toList(),
                BigDecimal.valueOf(endTime - startTime).divide(BigDecimal.valueOf(1_000_000), 4, RoundingMode.HALF_UP) // ns to ms
        );
    }
}
