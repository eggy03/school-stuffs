package io.github.eggy03.ml.smile.knn;

import lombok.extern.slf4j.Slf4j;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.io.Read;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

@Slf4j
@SuppressWarnings("java:S1192")
public class SmileKNNTesting {

    static void main() throws IOException, URISyntaxException {

        // smile k-nn expects numeric labels
        final Map<String, Integer> labelToID = Map.of(
                "Iris-setosa", 0,
                "Iris-versicolor", 1,
                "Iris-virginica", 2
        );

        // create a reverse map for analysis later
        final Map<Integer, String> idToLabel = Map.of(
                0, "Iris-setosa",
                1, "Iris-versicolor",
                2, "Iris-virginica"
        );

        // prepare training data
        DataFrame trainingDataFrame = Read.csv("src/main/resources/ml/knn/iris/csv/iris_training_dataset.csv", "header=true");
        int[] trainingDataLabels = trainingDataFrame
                .stream()
                .mapToInt(row -> labelToID.get(row.getString("species")))
                .toArray();

        double[][] trainingDataFeatures = trainingDataFrame
                .drop("species")
                .toArray();

        // prepare testing data
        DataFrame testingDataFrame = Read.csv("src/main/resources/ml/knn/iris/csv/iris_testing_dataset.csv", "header=true");
        int[] testingDataLabels = testingDataFrame
                .stream()
                .mapToInt(row -> labelToID.get(row.getString("species")))
                .toArray();

        double[][] testingDataFeatures = testingDataFrame
                .drop("species")
                .toArray();

        for (int k = 3; k < trainingDataFeatures.length; k = k+2) {

            int correctPredictions = 0;

            final KNN<double[]> trainedModel = KNN.fit(trainingDataFeatures, trainingDataLabels, k);

            for (int i = 0; i < testingDataFeatures.length; i++) {
                int actualLabel = testingDataLabels[i];
                int predictedLabel = trainedModel.predict(testingDataFeatures[i]);

                log.debug("Actual: {}, Predicted: {}", idToLabel.get(actualLabel), idToLabel.get(predictedLabel));

                if (predictedLabel == actualLabel)
                    correctPredictions++;
            }

            double accuracy = 100.00 * correctPredictions / testingDataFeatures.length;
            log.info("Total Training Samples: {} | Total Testing Samples: {} | k: {} | Correct Predictions: {} | Accuracy: {}%",
                    trainingDataFeatures.length, testingDataFeatures.length, k, correctPredictions, accuracy);
        }
    }
}
