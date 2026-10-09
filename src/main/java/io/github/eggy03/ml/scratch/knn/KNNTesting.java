package io.github.eggy03.ml.scratch.knn;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.eggy03.ml.scratch.knn.entity.Sample;
import io.github.eggy03.ml.scratch.knn.model.KNN;
import io.github.eggy03.util.JsonUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

@Slf4j
public class KNNTesting {

    static ObjectMapper objectMapper = new ObjectMapper();

    static List<Sample<String>> trainingData = JsonUtils.deserializeFromJson(
            "/ml/knn/iris/iris_training_dataset.json",
            objectMapper,
            objectMapper.getTypeFactory().constructParametricType(Sample.class, String.class)
    );

    static List<Sample<String>> testingData = JsonUtils.deserializeFromJson(
            "/ml/knn/iris/iris_testing_dataset.json",
            objectMapper,
            objectMapper.getTypeFactory().constructParametricType(Sample.class, String.class)
    );

    static void main() {

        for (int k = 3 ; k < trainingData.size() ; k = k+2) {

            final KNN<String> irisModel = new KNN<>(k, trainingData);
            final AtomicInteger correctPredictions = new AtomicInteger(0);

            testingData.forEach(testingSample -> {
                String actualLabel = testingSample.label();
                String trainingLabel = irisModel.predict(testingSample.featureVector());

                log.debug("For {} with features: {}, the prediction was: {}",
                        actualLabel, testingSample.featureVector().featureList(), trainingLabel
                );

                if (actualLabel.equals(trainingLabel))
                    correctPredictions.getAndIncrement();
            });

            double accuracy = 100.00 * correctPredictions.get() / testingData.size();
            log.info("Total Training Samples: {} | Total Testing Samples: {} | k: {} | Correct Predictions: {} | Accuracy: {}%",
                    trainingData.size(), testingData.size(), k, correctPredictions.get(), accuracy);
        }
    }
}
