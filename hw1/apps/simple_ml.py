"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

def readTrainXImage(image_filename):
    with gzip.open(image_filename,'rb') as f:
        # Read header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Error: magic number of input image file: {image_filename} is wrong.")
        num_of_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        # Read data
        # Pre-allocate the data array
        X = np.zeros((num_of_images, rows*cols), dtype=np.float32)

        # Read data for each image
        for i in range(num_of_images):
            #image_data = struct.unpack(f'>{rows * cols}B', f.read(rows * cols))
            # Read the exact number of bytes expected for one image
            image_bytes = f.read(rows * cols)
            if image_bytes is None:
                raise ValueError(f"Error: Read returned None for image {i}")
        
            # Debugging: Check if we got the correct number of bytes
            if len(image_bytes) != rows * cols:
                raise ValueError(f"Error reading image {i}: expected {rows * cols} bytes, got {len(image_bytes)} bytes.")            

            image_data = struct.unpack(f'>{rows * cols}B', image_bytes)
            X[i] = np.array(image_data, dtype=np.float32)
            X[i] /= 255
    return X, magic_number, num_of_images, rows, cols

def readTrainYLabel(label_filename):
    with gzip.open(label_filename,'rb') as f:
        # Read header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f"Error: magic number of input image file: {label_filename} is wrong.")
        num_of_items = struct.unpack('>I', f.read(4))[0]

        # Read data
        # Pre-allocate the data array
        y = np.zeros((num_of_items), dtype=np.uint8)

        for i in range(num_of_items):
            label_bytes = f.read(1)
            if label_bytes is None:
                raise ValueError(f"Error: Read returned None for item {i}")
        
            # Debugging: Check if we got the correct number of bytes
            if len(label_bytes) != 1:
                raise ValueError(f"Error reading item {i}: expected 1 bytes, got {len(label_bytes)} bytes.")            

            label_data = struct.unpack(f'>B', label_bytes)
            y[i] = np.array(label_data, dtype=np.uint8)[0]
    return y, magic_number, num_of_items

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    #raise NotImplementedError()

    X, magic_number_trainX, num_of_images, rows, cols = readTrainXImage(image_filename)
    y, magic_number_trainY, num_of_items = readTrainYLabel(label_filename) 

    '''
    print(f"magic_number_trainX = {magic_number_trainX}")
    print(f"num_of_images = {num_of_images}")
    print(f"rows = {rows}")
    print(f"cols = {cols}")
    print(f"X.dtype = {X.dtype}")
    print(f"X.shape = {X.shape}")
    print(f"X[2] = {X[2]}")
    print(f"magic_number_trainY = {magic_number_trainY}")
    print(f"num_of_items = {num_of_items}")
    print(f"y[:10] = {y[:10]}")
    print(f"y.shape = {y.shape}")
    '''
    return X, y    
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    pos_part = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    neg_part = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    neg_part_minus = ndl.mul_scalar(neg_part, -1)
    pos_neg_part = ndl.add(pos_part, neg_part_minus)
    pos_neg_sum = ndl.summation(pos_neg_part, axes=(0,))
    pos_neg_mean = ndl.divide_scalar(pos_neg_sum, Z.shape[0])

    '''
    print(f"")
    print(f"type(Z.shape[0]) = {type(Z.shape[0])}")
    print(f"mxk = {Z.shape[0]}x{Z.shape[1]}")
    print(f"Z.shape = {Z.shape}")
    print(f"y_one_hot.shape = {y_one_hot.shape}")
    print(f"pos_part.shape = {pos_part.shape}")
    print(f"neg_part.shape = {neg_part.shape}")
    print(f"neg_part_minus.shape = {neg_part_minus.shape}")
    print(f"pos_neg_part.shape = {pos_neg_part.shape}")
    print(f"pos_neg_sum.shape = {pos_neg_sum.shape}")
    print(f"pos_neg_mean.shape = {pos_neg_mean.shape}")
    print(f"type(pos_neg_mean) = {type(pos_neg_mean)}")
    print(f"pos_neg_mean = {pos_neg_mean}")
    '''

    return pos_neg_mean
    #raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
