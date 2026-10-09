package io.github.eggy03.ml.knn;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.eggy03.ml.knn.entity.FeatureVector;
import io.github.eggy03.ml.knn.entity.Sample;
import io.github.eggy03.ml.knn.model.KNN;
import io.github.eggy03.util.JsonUtils;
import lombok.extern.slf4j.Slf4j;

import java.util.List;

@Slf4j
public class KNNExample {

    static ObjectMapper objectMapper = new ObjectMapper();

    static List<Sample<String>> trainingData = JsonUtils.deserializeFromJson(
            "/ml/knn/iris/iris_training_dataset.json",
            objectMapper,
            objectMapper.getTypeFactory().constructParametricType(Sample.class, String.class)
    );

    static void main() {
        KNN<String> irisModel = new KNN<>(7, trainingData);
        log.info("Final Label Prediction: {}", irisModel.predict(new FeatureVector(List.of(3.1, 2.5, 2.4, 3.1))));
    }
}
