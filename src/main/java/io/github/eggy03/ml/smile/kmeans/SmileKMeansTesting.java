package io.github.eggy03.ml.smile.kmeans;

import lombok.extern.slf4j.Slf4j;
import smile.clustering.CentroidClustering;
import smile.clustering.KMeans;
import smile.data.DataFrame;
import smile.io.Read;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Map;

@Slf4j
public class SmileKMeansTesting {

    static void main() throws IOException, URISyntaxException {

        // smile expects numeric labels
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
        DataFrame trainingDataFrame = Read.csv("src/main/resources/ml/kmeans/iris/csv/iris.csv", "header=true");

        int[] trainingDataLabels = trainingDataFrame
                .stream()
                .mapToInt(row -> labelToID.get(row.getString("species")))
                .toArray();

        double[][] trainingDataFeatures = trainingDataFrame
                .drop("species")
                .toArray();

        CentroidClustering<double[], double[]> model = KMeans.fit(trainingDataFeatures, 3, 100);

        int[] predictedClusters = model.group();

        for (int i = 0; i < trainingDataFrame.nrow(); i++) {

            log.info("Sample: {} | Actual Label: {} | Predicted Cluster: {}",
                    i + 1,
                    idToLabel.get(trainingDataLabels[i]),
                    predictedClusters[i]
            );

        }
    }
}
