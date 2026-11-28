import tensorflow as tf
from pathlib import Path

model = tf.keras.Sequential([
    tf.keras.Input(shape=(30, 144)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(15, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

output_path = Path("evaluation/tests/dummy_model.keras")
model.save(output_path)

print(f"Dummy model saved at: {output_path}")