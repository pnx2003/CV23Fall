'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad


def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale
        4. beta: annealing temperature
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    
    energy_list = np.zeros((255,1))
    # get the size of the image
    img_height, img_width = img.shape


    # original pixel value
    original_pixel = img[loc[0], loc[1]]
    
    x, y= loc
    
    # TODO: calculate the energy
    for i in range(255):
        img[loc[0], loc[1]] = i
        

        new_weight = cal_pot(img[x, (y+1)%img_width] - img[x, y], norm) + cal_pot(img[x, y] - img[x, (y-1) % img_width], norm) + \
            cal_pot(img[(x+1)%img_height, y] - img[x,y], norm) + cal_pot(img[x ,y] - img[(x-1)%img_height ,y], norm)

        energy_list[i][0] = new_weight
        
    # normalize the energy
    energy_list = energy_list - energy_list.min()
    #energy_list = energy_list/energy_list.sum()



    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        # TODO
        inv_cdf = np.cumsum(probs)
        u = np.random.uniform()
        new_pixel = np.argmax(inv_cdf >= u)
        img[loc[0], loc[1]] = new_pixel

    except:
        raise ValueError(f'probs = {probs}')
    

    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''
    H, W = image.shape
    h, w = filter.shape
    padded_img = np.pad(image,((h//2, h//2), (w//2, w//2)), mode='wrap')
    image_tensor = torch.from_numpy(padded_img).unsqueeze(0).unsqueeze(0)
    filter_tensor = torch.from_numpy(filter).unsqueeze(0).unsqueeze(0)

    filtered_image = conv2d(image_tensor, filter_tensor, padding=0)
            
    return filtered_image.squeeze().numpy()

def main():
    # read the distorted image and mask image
    names = ["stone","sce","room"]
    sizes = ["small","big"]
    norms = ["L1","L2"]
    for name in names:
        for size in sizes:
            
            distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
            mask_path = f"../image/mask_{size}.bmp"
            ori_path = f"../image/{name}/{name}_ori.bmp"
                
            for norm in norms:
                if os.path.exists(f"gibbs_loss_{name}_{size}_{norm}.png"):
                    continue

                # read the BGR image
                distort = cv2.imread(distorted_path).astype(np.float64)
                mask = cv2.imread(mask_path).astype(np.float64)
                ori = cv2.imread(ori_path).astype(np.float64)

                # calculate initial energy
                red_channel = distort[:,:,2]
                energy = 0
                #calculate nabla_x
                filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
                energy += np.sum(cal_pot(filtered_img, norm), axis = (0,1))

                # calculate nabla_y
                filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
                energy += np.sum(cal_pot(filtered_img, norm), axis = (0,1))




                beta = np.linspace(1, 10, num=100)
                img_height, img_width, _ = distort.shape

                sweep = 100
                losses = []
                for s in tqdm(range(sweep)):
                    for i in range(img_height):
                        for j in range(img_width):
                            # only change the channel red
                            if mask[i,j,2] == 255:
                                distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta[s], norm)
                                
                    energy = 0
                    red_channel = distort[:, :, 2]
                    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
                    energy += np.sum(cal_pot(filtered_img, norm), axis=(0,1))
                    
                    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
                    energy += np.sum(cal_pot(filtered_img, norm), axis=(0,1))
                    
                    loss = ((distort[:,:,2] - ori[:,:,2])**2).sum()/(red_channel.shape[0]*red_channel.shape[1])
                    print(loss)
                    losses.append(loss)


                    save_path = f"./result/{name}/{size}/{norm}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
        
                import matplotlib.pyplot as plt
                plt.plot(list(range(0,100)), losses, label=f'Gibbs_sampler( {name}, {size}, {norm})')
                plt.legend()
                plt.xlabel('Sweep')
                plt.ylabel('Per Pixel Error')
                plt.savefig(f"gibbs_loss_{name}_{size}_{norm}.png")



if __name__ == "__main__":
    main()







        

