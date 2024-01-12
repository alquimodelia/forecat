import os

os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]

from forecat.models_definitions import MODELS_ARCHS, get_model_from_def

input_args = {
    "X_timeseries": 6, # Number of sentinel images
    "Y_timeseries": 1, # Number of volume maps
    "n_features_train": 12, # Number of sentinel bands
    "n_features_predict": 1, # We just want to predict the volume linearly
}
input_args = {
    "X_timeseries": 168, # Number of sentinel images
    "Y_timeseries": 24, # Number of volume maps
    "n_features_train": 18, # Number of sentinel bands
    "n_features_predict": 1, # We just want to predict the volume linearly
}
UNET = get_model_from_def("UNET", input_args=input_args,model_structures=MODELS_ARCHS,)
TRANSFORMER = get_model_from_def("Transformer", input_args=input_args,model_structures=MODELS_ARCHS,)

UNET.summary()
TRANSFORMER.summary()