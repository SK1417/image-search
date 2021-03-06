from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist 
import numpy as np 
import argparse 
import pickle 

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True)
ap.add_argument('-i', '--index', type=str, required=True) 
args = vars(ap.parse_args())

((trainX, _), (testX, _)) = mnist.load_data()

trainX = np.expand_dims(trainX, axis=-1)
trainX = trainX.astype('float32')/255.0

autoencoder = load_model(args['model'], compile=False)
print(autoencoder.summary())
encoder = Model(inputs=autoencoder.layers[1].input, outputs=autoencoder.layers[1].output) 
print(autoencoder.layers[1].input, autoencoder.input)

features = encoder.predict(trainX)
indexes = list(range(trainX.shape[0]))
data = {'indexes':indexes, 'features':features}

f = open(args['index'], 'wb')
f.write(pickle.dumps(data))
f.close() 