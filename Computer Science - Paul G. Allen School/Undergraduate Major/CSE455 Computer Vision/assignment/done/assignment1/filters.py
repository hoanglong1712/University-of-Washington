
import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    flip = np.flipud(np.fliplr(kernel))
    pad_height = Hk // 2
    pad_width = Wk // 2
    #padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))
    padded_image = zero_pad(image, pad_height, pad_width)
    for i in range (Hi):
        for j in range (Wi):
            roi = padded_image[i:i + Hk, j: j + Wk]
            for k in range (Hk):
                for l in range (Wk):
                    out[i,j] += roi[k,l] * flip[k,l]
                    pass
                pass
            #out[i,j] = np.sum(roi * flip)
            pass
        pass
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.zeros(((H+2*pad_height, W+2*pad_width)))
    for i in range(H):
        for j in range (W):
            out[i + pad_height, j + pad_width] = image[i, j]
            pass
        pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pad_height = Hk // 2
    pad_width = Wk // 2

    padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))
    flip = np.flipud(np.fliplr(kernel))

    for i in range (Hi):
        for j in range (Wi):
            roi = padded[i:i + Hk, j: j + Wk]            
            out[i,j] = np.sum(roi * flip)
            pass
        pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fh, fw = f.shape
    gh, gw = g.shape

    pad_height = gh // 2
    pad_width = gw // 2

    padded = np.pad(f, ((pad_height, pad_height), (pad_width, pad_width)))
    out = np.zeros((fh, fw))

    for i in range (fh):
        for j in range (fw):
            roi = padded[i:i + gh, j: j + gw]            
            out[i,j] = np.sum(roi * g)
            pass
        pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fh, fw = f.shape
    gh, gw = g.shape

    pad_height = gh // 2
    pad_width = gw // 2

    padded = np.pad(f, ((pad_height, pad_height), (pad_width, pad_width)))
    out = np.zeros((fh, fw))
    tmp = g - np.mean(g)
    
    for i in range (fh):
        for j in range (fw):
            roi = padded[i:i + gh, j: j + gw]            
            out[i,j] = np.sum(roi * tmp)  
            pass
        pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fh, fw = f.shape
    gh, gw = g.shape

    pad_height = gh // 2
    pad_width = gw // 2

    
    out = np.zeros((fh, fw))
    g_ = (g - np.mean(g)) / np.std(g)
    #f_ = f - np.mean(f) / np.std(f)
    padded = np.pad(f, ((pad_height, pad_height), (pad_width, pad_width)))
    
    for i in range (fh):
        for j in range (fw):
            # patch image at (n,m)
            roi = padded[i:i + gh, j: j + gw] 
            roi_ = (roi - np.mean(roi)) / np.std(roi)
            out[i,j] = np.sum(roi_ * g_)  
            pass
        pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ### END YOUR CODE

    return out
