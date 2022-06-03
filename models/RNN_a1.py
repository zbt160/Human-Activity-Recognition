from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape


# model.summary()



def init_model():
	# Initialize and compile this model
	model = Sequential()
	model.add(LSTM(50))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	return model