import numpy as np

def distribute_value(dz, shape):
    # """
    # Distributes the input value in the matrix of dimension shape
    
    # Arguments:
    # dz -- input scalar
    # shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    # Returns:
    # a -- Array of size (n_H, n_W) for which we distributed the value of dz
    # """

    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape

    # Compute the value to distribute on the matrix (≈1 line)
    average = dz/(n_H*n_W)

    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones((n_H, n_W)) * average
    ### END CODE HERE ###

    return a
