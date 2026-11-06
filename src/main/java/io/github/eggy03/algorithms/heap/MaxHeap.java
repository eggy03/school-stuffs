package io.github.eggy03.algorithms.heap;

import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Arrays;

/**
 * This class is NOT thread-safe
 */
@Slf4j
public class MaxHeap {

    @Getter
    private int @NonNull [] maxHeapArray;

    public MaxHeap(int @NonNull [] inputArray) {
        this.maxHeapArray = maxHeapify(inputArray);
    }

    private int @NonNull [] maxHeapify(int @NonNull [] inputArray) {

        // create an array copy
        int[] copy = Arrays.copyOf(inputArray, inputArray.length);

        if (inputArray.length == 0 || inputArray.length == 1)
            return copy;

        // browse backwards after skipping leaf nodes (n/2) - 1
        for (int i = (copy.length / 2) - 1; i >= 0; i--) { // O(n)

            // compare and swap parent with leaf nodes for satisfying max heap
            // repeat until leaf nodes are reached
            int parentNodeIndex = i;
            while (parentNodeIndex >= 0 && !isLeafNode(parentNodeIndex, copy)) { // theoretically O(n * log n)

                int parentNode = copy[parentNodeIndex];

                int leftChildIndex = (2 * parentNodeIndex) + 1;
                int rightChildIndex = (2 * parentNodeIndex) + 2;

                int leftChildNode = copy[leftChildIndex];
                int rightChildNode = rightChildIndex < copy.length ? copy[rightChildIndex] : Integer.MIN_VALUE; // right child won't exist in case of complete but non-full binary trees

                if (parentNode < leftChildNode || parentNode < rightChildNode) {

                    if (leftChildNode >= rightChildNode) {
                        copy[parentNodeIndex] = leftChildNode;
                        copy[leftChildIndex] = parentNode;
                        parentNodeIndex = leftChildIndex;
                    } else {
                        copy[parentNodeIndex] = rightChildNode;
                        copy[rightChildIndex] = parentNode;
                        parentNodeIndex = rightChildIndex;
                    }
                } else {
                    parentNodeIndex--; // if already max-heapified, shift back by one node upto root
                }
            }
        }

        return copy;
    }

    private boolean isLeafNode(int index, int @NonNull [] inputArray) {
        int maxIndex = inputArray.length - 1;
        return ((2 * index) + 1) > maxIndex; // note that this check is sufficient cause heap is a complete binary tree
    }

    /**
     * <p>
     * This insertion assumes that the binary tree is already max-heapified.
     * As such, it does not attempt to re-heapify it anymore.
     * It inserts the element at the last index and then compares itself with its ancestors
     * for its correct position.
     * </p>
     *
     * @param element the element to insert
     * @return the instance for chaining
     */
    public MaxHeap insert(int element) {
        // declare a copy with 1 extra space
        int[] maxHeapArrayCopy = Arrays.copyOf(maxHeapArray, maxHeapArray.length + 1);

        // copy the element at the last
        maxHeapArrayCopy[maxHeapArrayCopy.length - 1] = element;

        // traverse the parent nodes in order, upto the root node
        // and place the last element in the correct position

        int childNodeIndex = maxHeapArrayCopy.length - 1; // index of the newly inserted element
        int parentNodeIndex = childNodeIndex / 2; // start from the parent of the newly inserted element

        while (childNodeIndex > 0) {

            int parentNode = maxHeapArrayCopy[parentNodeIndex];
            int childNode = maxHeapArrayCopy[childNodeIndex];

            // swap parent with child if child is greater than parent
            if (parentNode < childNode) {
                maxHeapArrayCopy[parentNodeIndex] = childNode;
                maxHeapArrayCopy[childNodeIndex] = parentNode;
            }
            childNodeIndex = parentNodeIndex;
            parentNodeIndex = (parentNodeIndex - 1) / 2;
        }

        maxHeapArray = maxHeapArrayCopy;
        return this;
    }


    static void main() {
        int[] someArray = {50, 30, 20, 15, 10, 8, 16};
        log.info("\nGiven Array: {}\nMax-Heapified Array:{}",
                Arrays.toString(someArray),
                Arrays.toString(new MaxHeap(someArray)
                        .insert(60)
                        .getMaxHeapArray()
                )
        );
    }
}
