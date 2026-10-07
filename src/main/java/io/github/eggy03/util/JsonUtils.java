package io.github.eggy03.util;

import com.fasterxml.jackson.databind.JavaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.TypeFactory;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Unmodifiable;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Objects;

public final class JsonUtils {

    private JsonUtils() {
        throw new IllegalStateException("Utility Class");
    }

    @NotNull
    @Unmodifiable
    public static <S> List<S> deserializeFromJson(@NotNull String jsonPath, @NotNull ObjectMapper objectMapper, @NotNull JavaType objectType) {

        Objects.requireNonNull(jsonPath, "jsonPath cannot be null");
        Objects.requireNonNull(objectMapper, "objectMapper cannot be null");

        InputStream jsonStream = Objects.requireNonNull(
                JsonUtils.class.getResourceAsStream(jsonPath),
                "JSON resource " + jsonPath + " could not be located"
        );

        try (InputStreamReader jsonReader = new InputStreamReader(jsonStream)) {

            TypeFactory typeFactory = objectMapper.getTypeFactory();
            JavaType listType = typeFactory.constructCollectionType(List.class, objectType);

            List<S> result = objectMapper.readValue(jsonReader, listType);
            return result == null ? List.of() : result.stream().toList();

        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
