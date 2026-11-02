package io.github.eggy03.algorithms.search;

import lombok.experimental.UtilityClass;

import java.util.Arrays;

@UtilityClass
public class BinarySearch {

    private static int iterativeBinSearch(int element, int[] inputArray) {

        // create a copy of the array for thread safety
        int[] arrayCopy = Arrays.copyOf(inputArray, inputArray.length);

        int low = 0;
        int high = arrayCopy.length - 1;

        while (low <= high) {
            int mid = (high + low) / 2;

            if (element == arrayCopy[mid])
                return mid;

            if (element > arrayCopy[mid])
                low = mid + 1;

            if (element < arrayCopy[mid])
                high = mid - 1;
        }

        return -1;
    }
}
