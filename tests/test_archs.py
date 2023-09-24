from forecat.archs import CNNArch, DenseArch, LSTMArch, UNETArch

# import pytest


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


foreStackedCNN = CNNArch(**input_args)
foreStackedCNN_model = foreStackedCNN.architeture(block_repetition=8)
foreStackedCNN_model.summary()


foreStackedLSTM = LSTMArch(**input_args)
foreStackedLSTM_model = foreStackedLSTM.architeture(block_repetition=2)
foreStackedLSTM_model.summary()

foreUNET = UNETArch(**input_args)
foreUNET_model = foreUNET.architeture()
foreUNET_model.summary()



# import visualkeras

# visualkeras.layered_view(foreDense_model,legend=True, to_file='dense_model.png')#.show() # write and show
# visualkeras.layered_view(foreLSTM_model,legend=True, to_file='VanillaLSTM_model.png')#.show() # write and show
# visualkeras.layered_view(foreCNN_model, legend=True,to_file='VanillaCNN_model.png')#.show() # write and show
# visualkeras.layered_view(foreStackedCNN_model,legend=True, to_file='StackedCNN_model.png')#.show() # write and show
# visualkeras.layered_view(foreStackedLSTM_model, legend=True,to_file='StackedLSTM_model.png')#.show() # write and show
# visualkeras.layered_view(foreUNET_model,legend=True, to_file='UNET_model.png')#.show() # write and show