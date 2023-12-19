from keras_core import Sequential, layers, ops
from keras_core.activations import gelu

image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images


data_augmentation = Sequential(
    [
        layers.BatchNormalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)


def mlp(x, hidden_units, dropout_rate, activation=gelu):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.position_embedding.build(input_shape)

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = layers.Dense(units=projection_dim)(patch)
        encoded = projected_patches + layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


num_classes = 1
input_shape = (168, 18)


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 18
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 1
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier

output_shape = (24, 1)


def create_vit_classifier(
    input_layer,
    augmentation=False,
    projection_dim=18,
    activation_middle="relu",
    activation_end="relu",
    transformer_layers=1,
):
    inputs = input_layer
    if augmentation:
        # Augment data.
        augmented = data_augmentation(inputs)
        inputs = augmented
    if len(inputs.shape) > 3:
        # Create patches.
        patches = Patches(patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
        inputs = encoded_patches
    encoded_patches = inputs
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.BatchNormalization()(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.BatchNormalization()(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=transformer_units,
            dropout_rate=0.1,
            activation=activation_middle,
        )
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.BatchNormalization()(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(
        representation,
        hidden_units=mlp_head_units,
        dropout_rate=0.5,
        activation=activation_middle,
    )
    # TODO: make this a multipler of the time dimensiton (24)
    features = layers.Dense(24**2)(features)

    output_shape = (24, int(features.shape[-1] / 24))
    reshape = layers.Reshape(output_shape)(features)
    # Classify outputs.
    logits = layers.Dense(num_classes, activation=activation_end)(reshape)
    # # Create the Keras model.
    # model = Model(inputs=inputs, outputs=logits)
    return logits
