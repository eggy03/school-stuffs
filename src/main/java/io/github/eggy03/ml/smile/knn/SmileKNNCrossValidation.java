package io.github.eggy03.ml.smile.knn;

import lombok.extern.slf4j.Slf4j;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.io.Read;
import smile.validation.ClassificationValidations;
import smile.validation.CrossValidation;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

@Slf4j
public class SmileKNNCrossValidation {

    static void main() throws IOException, URISyntaxException {

        // smile expects numeric labels
        final Map<String, Integer> labelToID = Map.of(
                "Iris-setosa", 0,
                "Iris-versicolor", 1,
                "Iris-virginica", 2
        );

        // prepare training data
        DataFrame trainingDataFrame = Read.csv("src/main/resources/ml/knn/iris/csv/iris.csv", "header=true");

        int[] trainingDataLabels = trainingDataFrame
                .stream()
                .mapToInt(row -> labelToID.get(row.getString("species")))
                .toArray();

        double[][] trainingDataFeatures = trainingDataFrame
                .drop("species")
                .toArray();

        for (int k = 3; k < 120 ; k = k + 2) {
            final int kPrime = k;
            ClassificationValidations<KNN<double[]>> validation = CrossValidation.classification(10, trainingDataFeatures, trainingDataLabels, (x, y) -> KNN.fit(x, y, kPrime));
            log.info("K: {} | Accuracy: {}", kPrime, validation.avg().accuracy());
        }

    }
}
