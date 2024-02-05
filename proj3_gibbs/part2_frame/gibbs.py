''' 
This is file for part 1 
It defines the Gibbs sampler and we use cython for acceleration
'''
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.nn.functional import conv2d, pad

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
    
    #normalize the histogram
    
    histogram = histogram / (img_H * img_W)

    return histogram

def quick_conv(image, filter):
    h, w = filter.shape
    padded_img = np.pad(image, ((h//2, h//2),(w//2, w//2)),mode='wrap')  
    image_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0)
    filter_tensor = torch.from_numpy(filter).unsqueeze(0).unsqueeze(0)
    
    filtered_image = conv2d(image_tensor, filter_tensor, padding=0)

    return filtered_image.squeeze().numpy() 


def gibbs_sample(img_syn, hists_syn,
                 img_ori, hists_ori,
                 filter_list, sweep, bounds,
                 T, weight, num_bins):
    '''
    The gibbs sampler for synthesizing a texture image using annealing scheme
    Parameters:
        1. img_syn: the synthesized image, numpy array, shape: [H,W]
        2. hists_syn: the histograms of the synthesized image, numpy array, shape: [num_chosen_filters,num_bins]
        3. img_ori: the original image, numpy array, shape: [H,W]
        4. hists_ori: the histograms of the original image, numpy arrays, shape: [num_chosen_filters,num_bins]
        5. filter_list: the list of selected filters
        6. sweep: the number of sweeps
        7. bounds: the bounds of the responses of img_ori, a array of numpy arrays in shape [num_chosen_filters,2], bounds[x][0] max response, bounds[x][1] min response
        8. T: the initial temperature
        9. weight: the weight of the error, a numpy array in the shape of [num_bins]
        10. num_bins: the number of bins of histogram, a scalar
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
    '''

    H,W = (img_syn.shape[0],img_syn.shape[1])
    num_chosen_filters = len(filter_list)
    print(" ---- GIBBS SAMPLING ---- ")
    for s in tqdm(range(sweep)):
        for pos_h in range(H):
            for pos_w in range(W):
                pos = [pos_h, pos_w]
                img_syn,hists_syn = pos_gibbs_sample_update(img_syn,hists_syn,img_ori,hists_ori,filter_list,bounds,weight,pos,num_bins,T)
        max_error = (np.abs(hists_syn-hists_ori) @ weight).max()
        print(f'Gibbs iteration {s+1}: error = {(np.abs(hists_syn-hists_ori) @ weight).mean()} max_error: {max_error}')
        T = T * 0.8
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error} < 0.1, stop!")
            break
    return img_syn, hists_syn


def pos_gibbs_sample_update(img_syn, hists_syn,
                            img_ori, hists_ori,
                            filter_list, bounds,
                            weight, pos, 
                            num_bins, T):
    '''
    The gibbs sampler for synthesizing a value of single pixel
    Parameters:
        1. img_syn: the synthesized image, a numpy array in shape [H,W]
        2. hists_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        3. img_ori: the original image, a numpy array in shape [H,W]
        4. hists_ori: the histograms of the original image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        5. filter_list: the list of filters, a list of numpy arrays 
        6. bounds: the bounds of the responses of img_ori, a list of numpy arrays in shape [num_chosen_filters,2], in the form of (max_response, min_response)
        7. weight: the weight of the error, a numpy array in the shape of [num_bins]
        8. pos: the position of the pixel, a list of two scalars
        9. num_bins: the number of bins of histogram, a scalar
        10. T: current temperture of the annealing scheme
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
        hist_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
    '''

    H = img_syn.shape[0]
    W = img_syn.shape[1]
    pos_h = pos[0]
    pos_w = pos[1]
    
    hists = np.zeros(shape=(8, len(filter_list), num_bins))
    # calculate the conditional probability: p(I(i,j) = intensity | I(x,y),\forall (x,y) \neq (i,j))
    # perturb (i,j) pixel's intensity
    energy = np.zeros(num_bins)
    for i in range(8):
        img_syn[pos_h, pos_w] = i
        for idx in range(len(filter_list)):
           
            hists[i][idx] = np.array(get_histogram(quick_conv(img_syn, filter_list[idx]),num_bins , bounds[idx][0],bounds[idx][1], H, W))
            
    energy = np.sum(np.abs(hists - hists_ori) @ weight, axis=1)
    

    
    # normalize the energy
    energy = energy - energy.min()
    probs = np.exp(-energy/T)
    eps = 1e-10
    probs = probs + eps
    # normalize the probs
    probs = probs / probs.sum()
    
    # sample the intensity change the synthesized image
    try:
        inv_cdf = np.cumsum(probs)
        u = np.random.uniform()
        new_pixel = np.argmax(inv_cdf >= u)
        img_syn[pos[0], pos[1]] = new_pixel
    except:
        raise ValueError(f'probs = {probs}')

    # update the histograms
    hists_syn = hists[new_pixel]

    return img_syn, hists_syn


