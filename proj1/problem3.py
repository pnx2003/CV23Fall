'''
Thanks Yuran Xiang for the help of this problem
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
import numpy as np
import cv2
r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    # calculate the cdf: 1-1/r^2
    # calculate the inverse of cdf: (1/(1-k))^{1/2}
    
    return (1/(1-x))**(1/2)
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')
    for i in range(N):
        x0,y0 = points[i]
        angle = rad[i]
        x1 = int(x0 + length[i]*np.cos(angle))
        y1 = int(y0 + length[i]*np.sin(angle))
        if x1 > pixel:
            x1 = pixel - 1
        if y1 > pixel:
            y1 = pixel - 1
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if length[i] >= 1:
            cv2.line(bg, (x0, y0), (x1,y1), 0, 1)
        
    

    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''
    # Generating length
    length = GenLength(N)

    # Generating starting points uniformly
    points = np.random.randint(0,pixel,(N,2)) # Need to be changed
    # Generating orientation, range from 0 to 2\pi
    rad = np.random.uniform(0, 2*np.pi, (N,))# Need to be changed

    image = DrawLine(points,rad,length,pixel,N)
    return image,points,rad,length

def DownSampling(img,points,rad,length,pixel,N,rate):
    pixel = pixel//rate
    points = points//rate
    length = length/rate
    image = DrawLine(points, rad, length, pixel, N)
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''

    return image

def Drawcrop(image, i):
    x1 = np.random.randint(0, image.shape[0]-128)
    y1 = np.random.randint(0, image.shape[1]-128)
    croped_image = image[x1 : x1+128, y1:y1+128]
    cv2.imwrite('./pro3_result/croped_image_{}.png'.format(i),croped_image)
def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    Drawcrop(image1, 1)
    Drawcrop(image1, 2)
    Drawcrop(image2, 3)
    Drawcrop(image2, 4)
    Drawcrop(image3, 5)
    Drawcrop(image3, 6)
    
    
    return


def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
if __name__ == '__main__':
    main()
