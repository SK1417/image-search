from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist 
from imutils import build_montages 
import numpy as np 
import argparse 
import pickle 
import cv2 

def euclidean(a,b):
    return np.linalg.norm(a-b)

def perform_search(queryFeatures, index, maxResults=64):
    results = []
    for i in range(len(index['features'])):
        d = euclidean(queryFeatures, index['features'][i])
        results.append((d,i))
    
    results = sorted(results)[:maxResults]
    return results 


ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True)
ap.add_argument('-i', '--index', type=str, required=True) 
ap.add_argument('-s', '--samples', type=int, default=10)
args = vars(ap.parse_args())

((trainX, _), (testX, _)) = mnist.load_data()

trainX = np.expand_dims(trainX, axis=-1)
trainX = trainX.astype('float32')/255.0
testX = np.expand_dims(testX, axis=-1)
testX = testX.astype('float32')/255.0

autoencoder = load_model(args["model"])
index = pickle.loads(open(args["index"], "rb").read())

encoder = Model(inputs=autoencoder.layers[1].input, outputs=autoencoder.layers[1].output) 
features = encoder.predict(testX)

queryIndices = list(range(testX.shape[0]))
queryIndices = np.random.choice(queryIndices, size=args['samples'], replace=False)

for i in queryIndices:
    queryFeatures = features[i]
    results = perform_search(queryFeatures, index, maxResults=225)
    images = []

    for (d,j) in results:
        image = (trainX[j]*255).astype('uint8')
        image = np.dstack([image]*3)
        images.append(image)

    query = (testX[i]*255).astype('uint8')
    cv2.imshow('query', query)

    montage = build_montages(images, (28,28), (15,15))[0]
    cv2.imshow('montage', montage)
    cv2.waitKey(0)

    # python search.py -m "model.h5" -i "index.pickle"