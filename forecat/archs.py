import numpy as np
import tensorflow as tf
from keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
    Normalization,
    RepeatVector,
    Reshape,
    TimeDistributed,
)
from keras.models import Model

from forecat.utils import stays_if_not_bigger, stays_if_not_smaller


class ForeArch:
    """
    Base class for different architectures.

    Attributes:
    -----------
    X_timeseries: int
        Number of time series in the input
    Y_timeseries: int
        Number of time series in the output
    n_features_train: int
        Number of features in the training set
    n_features_predict: int
        Number of features in the prediction set
    dropout: float
        Dropout rate
    dense_out: int
        Number of output neurons in the dense layer
    input_shape: tuple
        Shape of the input layer
    output_shape: tuple
        Shape of the output layer
    """

    def __init__(
        self,
        X_timeseries,
        Y_timeseries,
        n_features_train,
        n_features_predict,
        dropout=0.35,
        activation_end="relu",
        activation_middle="relu",
    ):
        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.n_features_train = n_features_train
        self.n_features_predict = n_features_predict
        self.dense_out = self.Y_timeseries * self.n_features_predict
        self.set_input_shape()
        self.set_output_shape()
        self.dropout_value = dropout
        self.activation_end = activation_end
        self.activation_middle = activation_middle


    def set_input_shape(self):
        """Sets the input shape."""
        self.input_shape = (self.X_timeseries, self.n_features_train)

    def set_output_shape(self):
        """Sets the output shape."""
        self.output_shape = (self.Y_timeseries, self.n_features_predict)

    def interpretation_layers(self):
        pass

    def update_kernel(
        self,
        kernel,
        layer_shape,
        data_format="NHWC",  # TODO: mudar pa channels_first?
        kernel_dimension=-2,  # based on data format # ASsumir que isto esta sempre certo
    ):
        if isinstance(kernel_dimension, int):
            kernel_dimension = [kernel_dimension]
        if isinstance(kernel, int):
            kernel = [kernel]

        max_kernel_tuple_size = len(kernel_dimension)
        if len(kernel) < max_kernel_tuple_size:
            kernel += np.full(max_kernel_tuple_size - len(kernel), max(kernel))

        max_kernel_size = [layer_shape[i] for i in kernel_dimension]

        kernel = tuple(
            [
                stays_if_not_bigger(m, k)
                for m, k in zip(max_kernel_size, kernel)
            ]
        )
        # The `kernel_size` argument must be a tuple of 2 integers. Received: (1,)
        if len(kernel) == 1:
            kernel = kernel[0]
        return kernel

    def get_input_layer(
        self, normalization=True, bacth_norm=True, flatten_input=False
    ):
        """
        Returns the input layer with optional normalization and flattening.

        Parameters:
        -----------
        normalization: bool
            If True, applies normalization to the input layer.
        batch_norm: bool
            If True, applies batch normalization to the input layer.
        flatten_input: bool
            If True, applies flattening to the input layer.

        Returns:
        --------
        input_layer: keras.layer
            Input layer
        """
        input_layer = Input(self.input_shape)
        self.input_layer = input_layer
        if normalization:
            if bacth_norm:
                input_layer = BatchNormalization()(input_layer)
            else:
                input_layer = Normalization()(input_layer)

        if flatten_input:
            input_layer = Flatten()(input_layer)

        return input_layer

    def get_output_layer(self, out_layer, reshape_shape=None):
        """
        Returns the output layer with optional reshaping.

        Parameters:
        -----------
        out_layer: tensorflow.python.keras.engine.keras_tensor.KerasTensor
            Output layer
        reshape_shape: tuple
            Shape to reshape the output layer

        Returns:
        --------
        out_layer: tensorflow.python.keras.engine.keras_tensor.KerasTensor
            Output layer
        """
        if reshape_shape is None:
            reshape_shape = self.output_shape

        if not reshape_shape is None:
            out_layer = Reshape(reshape_shape)(out_layer)
        return out_layer

    def arch_block(self, input_layer):
        """
        Architecture block to be implemented in subclasses.
        """
        pass

    def stacked_repetition(
        self, input_layer, block_repetition=1, block_args={}
    ):
        args_use = block_args
        for i in range(block_repetition):
            if isinstance(block_args, list):
                args_use = block_args[i]
            input_layer = self.arch_block(input_layer, **args_use)

        return input_layer

    def paralel_repetition(self, input_layer, block_repetition=1, block=None, block_args=None):
        if not block:
            block = self.arch_block

        output_layer = list()
        block_in = input_layer
        block_args_in = block_args

        for i in range(block_repetition):
            if isinstance(input_layer, list):
                block_in = input_layer[i]
            if isinstance(block_args, list):
                block_args_in = block_args[i]
            output_layer.append(block(block_in, **block_args_in))

        return output_layer

    def multihead(
        self,
        input_layers,
        block_repetition=1,
        dim_split=None,
        reshape_into_original_dim_size=True,
        block=None,
    ):
        if isinstance(input_layers, list):
            block_repetition = len(input_layers)
            layer_shape = input_layers[0].shape
        else:
            layer_shape = input_layers.shape
        # if dim split not specified it will try to get the dimension
        # where to split
        if not dim_split:
            for i, dim in enumerate(layer_shape):
                if dim == block_repetition:
                    dim_split = i

        inputs = []
        for i in range(block_repetition):
            # If it comes as a list, get the index
            if isinstance(input_layers, list):
                x = tf.gather(input_layers, indices=i, axis=dim_split)
                if reshape_into_original_dim_size:
                    in_shape = (*x.shape[1:], 1)
                    x = Reshape(in_shape)(x)

            # Otherwise repeats the input_layer
            else:
                x = input_layers
            inputs.append(x)

        outputs = self.paralel_repetition(
            inputs, block_repetition=block_repetition, block=block
        )

        return outputs

    def architeture(self):
        pass


class DenseArch(ForeArch):
    """
    This architeture just follow the idea of a dense layers to solve the problem
    """

    def arch_block(
        self,
        x,
        dense_args={},
        filter_enlarger=4,
        filter_limit=200,
    ):
        if isinstance(dense_args, list):
            dense_args1 = dense_args[0]
            dense_args2 = dense_args[1]
        else:
            dense_args1 = dense_args
            dense_args2 = dense_args
        filters_out = dense_args2.pop("filters", None)
        if filters_out is  None:
            filters_out = self.dense_out

        x = Dense(
            stays_if_not_smaller(
                filters_out * filter_enlarger, filter_limit
            ),
            **dense_args1,
        )(x)
        x = Dense(filters_out, **dense_args2,)(x)

        return x

    def architeture(self):
        input_layer = self.get_input_layer(flatten_input=True)
        output_layer = self.arch_block(input_layer)
        output_layer = self.get_output_layer(output_layer)

        return Model(inputs=self.input_layer, outputs=output_layer)


class LSTMArch(ForeArch):
    """
    This architeture just follow the idea of a dense layers to solve the problem
    """

    def arch_block(
        self,
        x,
        lstm_args={"units": 50, "activation": "relu"},
        filter_enlarger=4,
        filter_limit=200,
    ):
        # Default LSTM arguments
        default_lstm_args = {"units": 50, "activation": "relu"}

        # If lstm_args is provided, update the default arguments
        if lstm_args is not None:
            default_lstm_args.update(lstm_args)

        # Apply LSTM layer
        x = LSTM(**default_lstm_args)(x)

        return x

    def interpretation_layers(self, output_layer, dense_args):
        output_layer = Dense(self.dense_out, **dense_args)(output_layer)
        output_layer = self.get_output_layer(output_layer)
        return output_layer

    def architeture(
        self,
        block_repetition=1,
        dense_args={"activation": "softplus"},
        block_args=None,
    ):
        input_layer = self.get_input_layer(flatten_input=False)
        if block_repetition == 1:
            output_layer = self.arch_block(input_layer)
        elif block_repetition > 1:
            if block_args is None:
                block_args = [{"lstm_args": {"return_sequences": True}}, {}]
            output_layer = self.stacked_repetition(
                input_layer, block_repetition, block_args=block_args
            )
        output_layer = Dropout(self.dropout_value)(output_layer)
        output_layer = self.interpretation_layers(output_layer, dense_args)

        return Model(inputs=self.input_layer, outputs=output_layer)


class CNNArch(ForeArch):
    """
    This architeture just follow the idea of a dense layers to solve the problem
    """

    def __init__(self, conv_dimension="1D", **kwargs):
        self.set_dimension_layer(conv_dimension)
        super().__init__(**kwargs)

    def set_dimension_layer(self, conv_dimension):
        if conv_dimension == "1D":
            self.MaxPooling = MaxPooling1D
            self.Conv = Conv1D
            self.Dropout = Dropout
        elif conv_dimension == "2D":
            self.MaxPooling = MaxPooling2D
            self.Conv = Conv2D
            self.Dropout = Dropout
        elif conv_dimension == "3D":
            self.MaxPooling = MaxPooling3D
            self.Conv = Conv3D
            self.Dropout = Dropout

    def arch_block(
        self,
        x,
        conv_args={},
        max_pool_args={"pool_size": 2},
        filter_enlarger=4,
        filter_limit=200,
    ):
        for k, v in {"filters": 16, "kernel_size": 3, "activation": "relu"}.items():
            if k not in conv_args:
                conv_args[k]=v

        x = self.Conv(**conv_args)(x)

        pool = self.update_kernel(max_pool_args["pool_size"], x.shape)
        max_pool_args.update({"pool_size": pool})

        x = self.MaxPooling(**max_pool_args)(x)
        x = self.Dropout(self.dropout_value)(x)

        return x

    def interpretation_layers(self, output_layer, dense_args=None, output_layer_args={}):
        if dense_args is None:
            dense_args = {}
            if self.activation_end != self.activation_middle:
                dense_args=[{"activation":self.activation_middle},
                {"activation":self.activation_end}
                ]
            else:
                dense_args = {"activation":self.activation_end}
            

        output_layer = DenseArch.arch_block(self, output_layer, dense_args=dense_args)
        output_layer = self.get_output_layer(output_layer, **output_layer_args)
        return output_layer

    def architeture(self, block_repetition=1, multitail=False):
        input_layer = self.get_input_layer(flatten_input=False)
        if block_repetition == 1:
            output_layer = self.arch_block(input_layer)
        elif block_repetition > 1:
            num_filters = [{"conv_args":{"filters":2**f}} for f in np.arange(block_repetition)]
            num_filters.reverse()
            output_layer = self.stacked_repetition(
                input_layer, block_repetition
            )
        output_layer = Flatten()(output_layer)

        if not multitail is False:
            if isinstance(multitail, list):
                multitail_repetition = len(multitail)
            elif isinstance(multitail, int):
                multitail_repetition = multitail
            else:
                multitail_repetition = 1


            output_layer = self.paralel_repetition(output_layer, multitail_repetition, self.interpretation_layers, block_args=multitail)
        else:
            output_layer = self.interpretation_layers(output_layer)



        return Model(inputs=self.input_layer, outputs=output_layer)


class UNETArch(ForeArch):
    """
    This architeture just follow the idea of a dense layers to solve the problem
    """

    def __init__(self, conv_dimension="1D", **kwargs):
        self.set_dimension_layer(conv_dimension)
        super().__init__(**kwargs)

    def set_dimension_layer(self, conv_dimension):
        if conv_dimension == "1D":
            self.MaxPooling = MaxPooling1D
            self.Conv = Conv1D
            self.Dropout = Dropout
        elif conv_dimension == "2D":
            self.MaxPooling = MaxPooling2D
            self.Conv = Conv2D
            self.Dropout = Dropout
        elif conv_dimension == "3D":
            self.MaxPooling = MaxPooling3D
            self.Conv = Conv3D
            self.Dropout = Dropout

    def architeture(self):
        from forecat.unet import AttResUNet1D

        n_filters = 16  # conv_args["filters"]

        model_UNET = AttResUNet1D(
            width=self.X_timeseries,
            num_bands=self.n_features_train,
            data_format="channels_last",
            n_filters=n_filters,
            num_classes=self.n_features_predict,
            activation_end="relu",
        )

        x = Model(
            inputs=model_UNET.input_layer, outputs=model_UNET.output_layer
        )

        return x


class EncoderDecoder(LSTMArch):
    def arch_block(
        self,
        x,
        lstm_args={"units": 50, "activation": "relu"},
        filter_enlarger=4,
        filter_limit=200,
    ):
        # Default LSTM arguments
        default_lstm_args = {"units": 50, "activation": "relu"}

        # If lstm_args is provided, update the default arguments
        if lstm_args is not None:
            default_lstm_args.update(lstm_args)

        x = LSTM(**lstm_args)(x)
        x = RepeatVector(self.Y_timeseries)(x)
        x = LSTM(**lstm_args, return_sequences=True)(x)
        x = Dropout(self.dropout_value)(x)

        return x

    def interpretation_layers(self, x, dense_args):
        time_dim_size = x.shape[1]
        x = TimeDistributed(
            Dense(self.dense_out / time_dim_size, **dense_args)
        )(x)
        x = Dropout(self.dropout_value)(x)
        x = self.get_output_layer(x)
        return x
