from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# Load VGG16
base_model = VGG16(include_top=True, weights="imagenet")

# (Optional) use penultimate layer only
vgg_model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)

# Save locally
vgg_model.save("models/vgg16_feature_extractor.keras")

print("✅ VGG16 saved successfully")
