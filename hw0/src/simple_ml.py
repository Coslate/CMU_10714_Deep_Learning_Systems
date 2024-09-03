import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass

def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE

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
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE

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
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    #pos_part = np.log(np.sum(np.exp(Z), axis=1))
    #neg_part = Z[np.arange(Z.shape[0]), y]
    #print(f"pos_part.shape = {pos_part.shape}")
    #print(f"neg_part.shape = {neg_part.shape}")
    #return np.mean(pos_part - neg_part)
    return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(Z.shape[0]), y])
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_iter = int(X.shape[0]/batch)
    for i in range(num_iter):
        tmpX = X[i*batch:(i+1)*batch, :]
        tmpY = y[i*batch:(i+1)*batch]
        Z = np.exp(tmpX@theta)
        sum_Z = np.reshape(np.sum(Z, axis=1), (Z.shape[0], 1))
        Z = Z/sum_Z
        Iy = np.zeros((batch, theta.shape[1]), dtype=int)
        Iy[np.arange(Iy.shape[0]), tmpY] = 1
        dtheta = (1/batch)*(tmpX.T@(Z-Iy))
        theta += -lr*dtheta
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    pass
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
