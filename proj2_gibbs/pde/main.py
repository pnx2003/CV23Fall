'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os




def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''

    x,y = loc
    img_height, img_width = img.shape
    laplacian = (img[(x-1)%img_height, y] + img[(x+1)%img_height, y] \
        + img[x, (y-1)%img_width] + img[x, (y+1)%img_width] - 4 * img[x, y])
    
    img[x,y] = img[x,y] + 0.1*beta*laplacian
    

    return img


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
                if os.path.exists(f"pde_loss_{name}_{size}_{norm}.png"):
                    continue

    


                # read the BGR image
                distort = cv2.imread(distorted_path).astype(np.float64)
                mask = cv2.imread(mask_path).astype(np.float64)
                ori = cv2.imread(ori_path).astype(np.float64)



                beta = 1
                img_height, img_width, _ = distort.shape

                sweep = 100
                losses = []
                for s in tqdm(range(sweep)):
                    for i in range(img_height):
                        for j in range(img_width):
                            # only change the channel red
                            if mask[i,j,2] == 255:
                                distort[:,:,2] = pde(distort[:,:,2], [i,j], beta)

                    # TODO

                    if s % 10 == 0:
                        save_path = f"./result/{name}/{size}/{norm}"
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)
                    
                    loss = ((distort[:,:,2] - ori[:,:,2])**2).sum()/(distort.shape[0]*distort.shape[1])
                    losses.append(loss)
            
    
                import matplotlib.pyplot as plt
                plt.plot(list(range(0,100)), losses, label=f'PDE update({name}, {size}, {norm})')
                plt.legend()
                plt.xlabel('Sweep')
                plt.ylabel(f'Per Pixel Error')
                plt.savefig(f"pde_loss_{name}_{size}_{norm}.png")

if __name__ == "__main__":
    main()







        

