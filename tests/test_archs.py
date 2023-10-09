

import pytest

from forecat.archs import CNNArch, DenseArch, EncoderDecoder, LSTMArch


def make_model_see_summary():
    X_timeseries = 168
    Y_timeseries = 24
    n_features_train = 18
    n_features_predict = 1

    input_args = {
        "X_timeseries": X_timeseries,
        "Y_timeseries": Y_timeseries,
        "n_features_train": n_features_train,
        "n_features_predict": n_features_predict,
    }

    foreDense = DenseArch(**input_args)
    foreDense_model = foreDense.architeture()
    foreDense_model.summary()

    foreLSTM = LSTMArch(**input_args)
    foreLSTM_model = foreLSTM.architeture()
    foreLSTM_model.summary()

    foreCNN = CNNArch(**input_args)
    foreCNN_model = foreCNN.architeture()
    foreCNN_model.summary()

    foreStackedCNN = CNNArch(**input_args)
    foreStackedCNN_model = foreStackedCNN.architeture(block_repetition=2)
    foreStackedCNN_model.summary()

    foreStackedLSTM = LSTMArch(**input_args)
    foreStackedLSTM_model = foreStackedLSTM.architeture(block_repetition=2)
    foreStackedLSTM_model.summary()

    foreEncoderDecoder = EncoderDecoder(**input_args)
    foreEncoderDecoder_model = foreEncoderDecoder.architeture()
    foreEncoderDecoder_model.summary()


@pytest.mark.parametrize("backend", ["tensorflow", "jax", "torch"])
def test_model_making(backend):

    #os.environ["KERAS_BACKEND"] = backend
    #from forecat.archs import CNNArch, DenseArch, 
    #   LSTMArch, UNETArch, EncoderDecoder

    # TODO: try to test with all the different backedn
    make_model_see_summary()    
