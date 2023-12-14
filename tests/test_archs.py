import keras_core as keras
import numpy as np
import pytest

import forecat

# The model structures
model_structures = {
    "VanillaCNN": {"arch": "CNNArch"},
    "VanillaDense": {"arch": "DenseArch"},
    "VanillaLSTM": {"arch": "LSTMArch"},
    "StackedCNN": {"arch": "CNNArch", "architecture_args": {"block_repetition": 2}},
    "StackedDense": {"arch": "DenseArch", "architecture_args": {"block_repetition": 2}},
    "StackedLSTM": {"arch": "LSTMArch", "architecture_args": {"block_repetition": 2}},
    "UNET": {"arch": "UNETArch"},
    "EncoderDecoder": {"arch": "EncoderDecoder"},
    "Transformer": {"arch": "Transformer"},
}

# The input arguments
input_args = {
    "X_timeseries": 168,
    "Y_timeseries": 24,
    "n_features_train": 18,
    "n_features_predict": 1,
}

# The compile arguments
compile_args = {
    "optimizer": "adam",
    "loss": "mse",
}


# The fixture for model name
@pytest.fixture(params=model_structures.keys())
def model_name(request):
    return request.param


# The test function
def test_model(model_name):
    # Clear the session
    keras.backend.clear_session()

    # Get the model configuration
    model_conf = model_structures[model_name]
    architecture_args = model_conf.get("architecture_args", {})

    # Create the architecture and model
    architecture_class = getattr(forecat.archs, model_conf["arch"])
    forearch = architecture_class(**input_args)
    foremodel = forearch.architecture(**architecture_args)
    foremodel.summary()
    foremodel.compile(**compile_args)

    # Create the test data
    input_shape = (2, *foremodel.input_shape[1:])
    output_shape = (2, *foremodel.output_shape[1:])
    X_test = np.ones(input_shape)
    Y_test = np.ones(output_shape)

    # Fit the model
    foremodel.fit(X_test, Y_test)
