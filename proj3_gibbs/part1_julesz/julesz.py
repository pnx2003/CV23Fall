''' 
This is the main file of Part 1: Julesz Ensemble
'''

from numpy.ctypeslib import ndpointer
import numpy as np
from filters import get_filters
import cv2
from torch.nn.functional import conv2d, pad
import torch
from gibbs import gibbs_sample

def conv(image, filter):
    
    ''' 
    Computes the filter response on an image.
    Notice: Choose your padding method!
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape (x, x)
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    H, W = image.shape
    h, w = filter.shape
    padded_img = np.pad(image, ((h//2, h//2),(w//2, w//2)),mode='wrap')
    image_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0)
    filter_tensor = torch.from_numpy(filter).unsqueeze(0).unsqueeze(0)
    
    filtered_image = conv2d(image_tensor, filter_tensor, padding=0)

    return filtered_image.squeeze().numpy()

def get_histogram(filtered_image,bins_num, max_response, min_response, img_H, img_W):
    ''' 
    Computes the normalized filter response histogram on an image.
    Parameters:
        1. filtered_image: numpy array of shape (H, W)
        2. bins_num: int, the number of bins
        3. max_response: int, the maximum response of the filter
        4. min_response: int, the minimum response of the filter
    Return:
        1. histogram: histogram (numpy array)
    '''

    histogram, _ = np.histogram(filtered_image, bins=bins_num, range=(min_response, max_response))
    import pdb
    pdb.set_trace()
    #normalize the histogram
    histogram = histogram / (img_H * img_W)

    return np.array(histogram).astype(np.float32)

def julesz(img_size = 64, img_name = "fur_obs.jpg", save_img = True):
    ''' 
    The main method
    Parameters:
        1. img_size: int, the size of the image
        2. img_name: str, the name of the image
        3. save_img: bool, whether to save intermediate results, for autograder
    '''
    max_intensity = 255

    # get filters
    F_list = get_filters()
    F_list = [filter.astype(np.float32) for filter in F_list]

    # selected filter list, initially set as empty
    filter_list = []
    filter_idx = []


    # size of image
    img_H  = img_W = img_size


    # read image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity 

    # store the original image
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))

    # synthesize image from random noise
    img_syn = np.random.randint(0, 8, img_ori.shape).astype(np.float32)
    filtered_syn = np.array([conv(img_syn, filter) for filter in F_list])
    filtered_ori = np.array([conv(img_ori, filter) for filter in F_list])
    weights = np.array([10,7,5,3,2.5,2,1.5,1,1.5,2,2.5,3,5,7,10], dtype=np.float32)
    bounds = np.array([np.array([max(map(max, filtered)),min(map(min,filtered))]) for filtered in filtered_ori])
    hist_ori = np.array([get_histogram(filtered_ori[i], 15, bounds[i][0], \
            bounds[i][1], img_H, img_W) for i in range(len(filtered_ori))])
    hist_syn = np.array([get_histogram(filtered_syn[i], 15, bounds[i][0], \
            bounds[i][1], img_H, img_W) for i in range(len(filtered_syn))])
    
    error = np.abs(hist_ori - hist_syn) @ weights
    threshold = 0.1 # TODO
    sweep = 50
    max_error = max(error)
    round = 0
    T = 0.01
    print("---- Julesz Ensemble Synthesis ----")
    while max_error > threshold: # Not empty
        idx = error.argmax()
        filter_idx.append(idx)
        filter_list.append(F_list[idx])

        img_syn, _ = gibbs_sample(img_syn, hist_syn[filter_idx], img_ori, hist_ori[filter_idx], \
            filter_list, sweep, list(bounds[filter_idx]), T, weights, 15)
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)
        round += 1
        filtered_ori = np.array([conv(img_ori, filter) for filter in F_list])
        filtered_syn = np.array([conv(img_syn, filter) for filter in F_list])
        hist_syn = np.array([get_histogram(filtered_syn[i], 15, bounds[i][0], \
            bounds[i][1], img_H, img_W) for i in range(len(F_list))])
        error = np.abs(hist_ori - hist_syn) @ weights
        total_error = error.mean()
        with open (f"results/{img_name.split('.')[0]}.txt", 'a') as f:
            f.write(str(total_error))
            f.write('\n')
        max_error = max(error)

    return img_syn  # return for testing
    
if __name__ == '__main__':
    julesz()
