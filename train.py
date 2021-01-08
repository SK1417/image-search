import matplotlib
matplotlib.use('Agg')

from autoencoders import Autoencoder
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.datasets import mnist 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
import cv2 

def visualize_preds(decoded, gt, samples=10):
    outputs = None 

    for i in range(samples):
        original = (gt[i]*255).astype('uint8')
        recon = (decoded[i]*255).astype('uint8')

        output = np.hstack([original, recon])

        if outputs is None:
            outputs = output
        else:
            outputs = np.vstack([outputs, output])
        
    return outputs

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True)
ap.add_argument('-o', '--output', type=str, default='output.png')
ap.add_argument('-p', '--plot', type=str, default='plot.png')
args = vars(ap.parse_args())

EPOCHS = 20
INIT_LR = 1e-3
BS = 32

print('Loadin MNIST...')
((trainX, _), (testX, _)) = mnist.load_data()

trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype('float32')/255.0
testX = testX.astype('float32')/255.0

(encoder, decoder, autoencoder) = Autoencoder.build(28, 28, 1)
opt = Adam(lr=1e-3, decay=INIT_LR/EPOCHS)
autoencoder.compile(optimizer=opt, loss='mse')

H = autoencoder.fit(
    trainX,
    trainX,
    validation_data = (testX, testX),
    epochs = EPOCHS,
    batch_size = BS 
)

decoded = autoencoder.predict(testX)
vis = visualize_preds(decoded, testX)
cv2.imwrite(args['output'], vis)

N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure() 
plt.plot(N, H.history['loss'], label='training_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.title('training_loss_and_accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Acc')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

autoencoder.save(args['model'], save_format='h5')
