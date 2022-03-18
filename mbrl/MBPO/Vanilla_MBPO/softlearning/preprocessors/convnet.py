import tensorflow as tf

from softlearning.models.feedforward import feedforward_model
from softlearning.utils.keras import PicklableKerasModel


def convnet_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        conv_filters=(32, 32),
        conv_kernel_sizes=((5, 5), (5, 5)),
        pool_type='MaxPool2D',
        pool_sizes=((2, 2), (2, 2)),
        pool_strides=(2, 2),
        dense_hidden_layer_sizes=(64, 64),
        data_format='channels_last',
        name="convnet_preprocessor",
        make_picklable=True,
        *args,
        **kwargs):
    if data_format == 'channels_last':
        H, W, C = image_shape
    elif data_format == 'channels_first':
        C, H, W = image_shape

    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    images_flat, input_raw = tf.keras.layers.Lambda(
        lambda x: [x[..., :H * W * C], x[..., H * W * C:]]
    )(concatenated_input)

    images = tf.keras.layers.Reshape(image_shape)(images_flat)

    conv_out = images
    for filters, kernel_size, pool_size, strides in zip(
            conv_filters, conv_kernel_sizes, pool_sizes, pool_strides):
        conv_out = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="SAME",
            activation=tf.nn.relu,
            *args,
            **kwargs
        )(conv_out)
        conv_out = getattr(tf.keras.layers, pool_type)(
            pool_size=pool_size, strides=strides
        )(conv_out)

    flattened = tf.keras.layers.Flatten()(conv_out)
    concatenated_output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([flattened, input_raw])

    output = (
        feedforward_model(
            input_shapes=(concatenated_output.shape[1:].as_list(), ),
            output_size=output_size,
            hidden_layer_sizes=dense_hidden_layer_sizes,
            activation='relu',
            output_activation='linear',
            *args,
            **kwargs
        )([concatenated_output])
        if dense_hidden_layer_sizes
        else concatenated_output)

    model = PicklableKerasModel(inputs, output, name=name)

    return model
