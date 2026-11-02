package io.github.eggy03.algorithms.sort;

import io.github.eggy03.algorithms.entity.ArraySortResult;
import io.github.eggy03.algorithms.exceptions.AlreadySortedException;
import lombok.NonNull;
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

    public static int @NotNull [] recursiveAscendingInsertionSort(int @NonNull [] inputArray, int position) {

        if (inputArray.length == 0)
            throw new AlreadySortedException("Array to be sorted cannot be empty");

        if (inputArray.length == 1)
            return inputArray;

        // position here is equivalent to j in the iterative version
        // and j starts with 1 because i = j-1
        if (position < 1)
            throw new IllegalStateException("Position cannot be less than 1");

        // base case
        // if position has reached the end of the array, stop
        if (position > inputArray.length - 1)
            return inputArray;

        // create a copy of the array for thread safety
        int[] arrayCopy = Arrays.copyOf(inputArray, inputArray.length);

        int i = position - 1;
        int lastElement = arrayCopy[position];

        while (i >= 0 && arrayCopy[i] > lastElement) {
            arrayCopy[i + 1] = arrayCopy[i];
            i--;
        }
        arrayCopy[i + 1] = lastElement;

        // increase the position and recursively call
        return recursiveAscendingInsertionSort(arrayCopy, position + 1 );
    }
}
