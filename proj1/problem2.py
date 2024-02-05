'''
This is the code for project 1 question 2
Question 2: Verify the 1/f power law observation in natural images in Set A
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
path = "./image_set/setA/"
colorlist = ['red', 'blue', 'black', 'green']
linetype = ['-', '-', '-', '-']
labellist = ["natural_scene_1.jpg", "natural_scene_2.jpg",
                 "natural_scene_3.jpg", "natural_scene_4.jpg"]

img_list = [cv2.imread(os.path.join(path,labellist[i]), cv2.IMREAD_GRAYSCALE) for i in range(4)]
def fft(img):
    ''' 
    Conduct FFT to the image and move the dc component to the center of the spectrum
    Tips: dc component is the one without frequency. Google it!
    Parameters:
        1. img: the original image
    Return:
        1. fshift: image after fft and dc shift
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    return fshift

def amplitude(fshift):
    '''
    Parameters:
        1. fshift: image after fft and dc shift
    Return:
        1. A: the amplitude of each complex number
    '''

    A = np.abs(fshift) # Need to be changed

    return A

def xy2r(x, y, centerx, centery):
    ''' 
    change the x,y coordinate to r coordinate
    '''
    rho = math.sqrt((x - centerx)**2 + (y - centery)**2)
    return rho

def cart2porl(A,img):
    ''' 
    Parameters: 
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. f: the frequency list 
        2. A2_f: the amplitude of each frequency
    Tips: 
        1. Use the function xy2r to get the r coordinate!
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    # build the r coordinate
    basic_f = 1
    max_r = min(centerx,centery)
    # the frequency coordinate
    f = np.arange(0,max_r + 1,basic_f)

    # the following process is to do the sampling for each frequency of f
    A_f = np.ones_like(f)
    distances = np.sqrt((np.arange(img.shape[0])[:, np.newaxis] - centerx)**2 + (np.arange(img.shape[1]) - centery)**2)
    for i in range(len(f)):
        indices = np.where(np.round(distances) == f[i])
        count = np.sum(A[indices])/(len(indices[0]))
        A_f[i] = count

    return f, A_f



def get_S_f0(A,img):
    ''' 
    Parameters:
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. S_f0: the S(f0) list
        2. f0: frequency list
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    
    f, A_f = cart2porl(A,img)
    S_f0 = np.zeros(100) # Need to be changed
    f0 = np.arange(0,100,1) # Need to be changed
    # distances = np.sqrt((np.arange(img.shape[0])[:, np.newaxis] - centerx)**2 + (np.arange(img.shape[1]) - centery)**2)
    for i in range(len(f0)):
        
        sum_A = np.sum([A_f[x]**2*x for x in range(f0[i],2*f0[i]+1)])
        S_f0[i] = sum_A
    return S_f0, f0
    
def main():
    plt.figure(1)
    # q1
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        f, A_f = cart2porl(A,img_list[i])
        plt.plot(np.log(f[1:190]),np.log(A_f[1:190]), color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("1/f law")
    plt.savefig("./pro2_result/f1_law.jpg", bbox_inches='tight', pad_inches=0.0)
    plt.figure(2)
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        S_f0, f0 = get_S_f0(A,img_list[i])
        plt.plot(f0[10:],S_f0[10:], color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("S(f0)")
    plt.savefig("./pro2_result/S_f0.jpg", bbox_inches='tight', pad_inches=0.0)
if __name__ == '__main__':
    main()
