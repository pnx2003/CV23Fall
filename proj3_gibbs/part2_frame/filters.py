''' 
This file is part of the code for Part 1:
    It contains a function get_filters(), which generates a set of filters in the
format of matrices. (Hint: You add more filters, like the Dirac delta function, whose response is the intensity of the pixel itself.)
'''
import numpy as np
import math


def get_filters():
    '''
    define set of filters which are in the form of matrices
    Return
          F: a list of filters

    '''

    # nabla_x, and nabla_y
    F = [np.array([-1, 0, 1]).reshape((1, 3)), np.array([-1, 0, 1]).reshape((3, 1))]
    # gabor filter
    F += [gabor for size in [3,5] for theta in range(0, 150, 30)  for gabor in gaborFilter(size, theta)]
    F += [np.array([1]).reshape((1,1))]
    

    return F


def gaborFilter(size, orientation):
    """
      [Cosine, Sine] = gaborfilter(scale, orientation)

      Defintion of "scale": the sigma of short-gaussian-kernel used in gabor.
      Each pixel corresponds to one unit of length.
      The size of the filter is a square of size n by n.
      where n is an odd number that is larger than scale * 6 * 2.
    """

    assert size % 2 != 0

    halfsize = math.ceil(size / 2)
    theta = (math.pi * orientation) / 180
    Cosine = np.zeros((size, size))
    Sine = np.zeros((size, size))
    gauss = np.zeros((size, size))
    scale = size / 6

    for i in range(size):
        for j in range(size):
            x = ((halfsize - (i+1)) * np.cos(theta) + (halfsize-(j+1)) * np.sin(theta)) / scale
            y = (((i+1) - halfsize) * np.sin(theta) + (halfsize-(j+1)) * np.cos(theta)) / scale

            gauss[i, j] = np.exp(-(x**2 + y**2/4) / 2)
            Cosine[i, j] = gauss[i, j] * np.cos(2*x)
            Sine[i, j] = gauss[i, j] * np.sin(2*x)

    k = np.sum(np.sum(Cosine)) / np.sum(np.sum(gauss))
    Cosine = Cosine - k * gauss
    return Cosine, Sine

if __name__ == '__main__':
    get_filters()



