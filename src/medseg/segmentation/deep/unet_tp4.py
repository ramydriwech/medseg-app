"""U-Net architecture of TP4, copied verbatim from ``TP4_IM.ipynb`` (cell 7).

Encoder: 5 blocks Conv2D -> Dropout -> Conv2D with 16, 32, 64, 128, 256 filters (3x3, ReLU,
he_normal, same padding), MaxPooling 2x2 between blocks; dropout 0.1/0.1/0.2/0.2/0.3.
Decoder: 4 blocks Conv2DTranspose(2x2, stride 2) + skip concatenation + conv block.
Output: Conv2D(1, 1x1, sigmoid). 1 941 105 trainable parameters for 256x256x3 inputs.

TensorFlow is imported lazily so that the rest of the package works without it.
"""

from __future__ import annotations


def simple_unet_model(img_height: int, img_width: int, img_channels: int):
    """Build the TP4 U-Net (Keras functional API)."""
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, Input, MaxPooling2D, concatenate
    from tensorflow.keras.models import Model

    inputs = Input(shape=(img_height, img_width, img_channels))

    def conv_block(x, filters, dropout_rate):
        x = Conv2D(filters, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(filters, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
        return x

    # Encoder: 16 -> 32 -> 64 -> 128 -> 256 filters.
    c1 = conv_block(inputs, 16, 0.1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 32, 0.1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 64, 0.2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 128, 0.2)
    p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 256, 0.3)

    # Decoder: 128 -> 64 -> 32 -> 16 filters with skip connections.
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    c6 = conv_block(concatenate([u6, c4]), 128, 0.2)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    c7 = conv_block(concatenate([u7, c3]), 64, 0.2)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
    c8 = conv_block(concatenate([u8, c2]), 32, 0.1)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
    c9 = conv_block(concatenate([u9, c1]), 16, 0.1)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return Model(inputs=inputs, outputs=outputs, name="Simple_UNet")
