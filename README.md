# image-search

Credits to Adrian Rosebrock and Pyimagesearch for their tutorial. It helped me develop my own autoencoder for the cifar10 dataset after some trial and error. 

Refer: https://www.pyimagesearch.com/2020/03/30/autoencoders-for-content-based-image-retrieval-with-keras-and-tensorflow/

The mnist/ folder contains the files required to run the model on the mnist dataset. The cifar/ folder contains the same for the cifar10 dataset. 

* The autoencoder.py python file is for the model class. You can modify the model as you wish here. 
* The train.py/train_cifar/py file is for building and training the model. You can pass in the filters for each layer, and the model will be built based on this. Running this file requires '-m' save path for model, '-o' save path for output image, '-p' save path for error plot.
* The index.py/index_cifar.py file is for making the index of all the training images in the dataset. Requires "-m" path to saved model "-i" path to save index pickle file
* The search.py/search_cifar.py file is for the actual CBIR application. Requires "-m": path to model file, "-i": path to index file, "-s": number of sample results to show (default=5)
