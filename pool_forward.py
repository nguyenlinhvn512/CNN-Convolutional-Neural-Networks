import numpy as np

def pool_forward(A_prev, hparameters, mode="max"):
    # """
    # Implements the forward pass of the pooling layer
    
    # Arguments:
    # A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    # hparameters -- python dictionary containing "f" and "stride"
    # mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    # Returns:
    # A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    # cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    # """

    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]

    # Define the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))

    ### START CODE HERE ###
    for i in range(m):                         # loop over the training examples
        # loop on the vertical axis of the output volume
        for h in range(n_H):
            # Find the vertical start and end of the current "slice" (≈2 lines)
            vert_start = h*stride
            vert_end = h*stride + f

            # loop on the horizontal axis of the output volume
            for w in range(n_W):
                # Find the vertical start and end of the current "slice" (≈2 lines)
                horiz_start = w*stride
                horiz_end = w*stride + f

                for c in range(n_C):            # loop over the channels of the output volume

                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]

                    # Compute the pooling operation on the slice.
                    # Use an if statement to differentiate the modes.
                    # Use np.max and np.mean.
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    ### END CODE HERE ###

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)

    # Making sure your output shape is correct
    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache
