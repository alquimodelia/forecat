# Forecat

Forecat is a Python package that provides a Keras-based forecast model builder.

## Timeseries Forecasting 

## Making Predictions with Keras Models

Forecat allows you to make predictions using the trained models. Once a model is trained, you can use the `model.predict()` method to make predictions on new data instances. For example, you can pass a new input data `X` to the trained model and get the predicted output `yhat` using the following code:


Forecat also supports saving and loading trained models using the `SavedModel` format or the HDF5 format, which allows you to reuse the models for future predictions[[2]](https://stackoverflow.com/questions/31914161/how-to-convert-rmd-into-md-in-r-studio)[[8]](https://docs.readme.com/rdmd/docs).

## Keras Model Components

When working with Keras models, it's important to understand the different components that make up a model:

- Architecture/Configuration: Specifies the layers and their connectivity in the model.
- Weights: The trainable parameters in the model that influence the output.
- Optimizer: The optimizer/loss function used to minimize the loss during training.
- Set of Losses and Metrics: The losses and metrics that are compiled with the model using the `model.compile()` method.

Forecat provides a high-level interface to create, configure, and train models, taking care of these components behind the scenes[[2]](https://stackoverflow.com/questions/31914161/how-to-convert-rmd-into-md-in-r-studio).

## Usage

To use Forecat, follow these steps:

1. Install Poetry for dependency management.
2. Install Forecat and its dependencies using Poetry: `poetry install`.
3. Use the `poetry run` command to run your scripts: `poetry run python your_script.py`.
4. Use the provided API to build and train forecast models using Keras and TensorFlow.
5. Make predictions using the trained models.

## Contribution

Contributions to Forecat are welcome! If you find any issues or have suggestions for improvement, please feel free to contribute. Make sure to update tests as appropriate and follow the contribution guidelines.

## License

Forecat is licensed under the MIT License, which allows you to use, modify, and distribute the package according to the terms of the license. For more details, please refer to the [LICENSE](LICENSE) file.
