""" import necessary libs """
import numpy as np
import cv2
from P1_q1 import conv2

""" function to blend images """
def imgBlending(l_fimg, l_bimg, gMask, layers):
    
    """ blend images """
    LS = []
    for la,lb,mask in zip(l_fimg,l_bimg,gMask):
        ls = la * mask + lb * (1 - mask)
        LS.append(np.float32(ls))

    """ reconstruct final image """
    lap_bl = LS[0]
    for i in range(1,layers):
        lap_bl = conv2(upSampler(lap_bl,LS[i]),w)
        lap_bl = cv2.add(lap_bl,LS[i])
        
    final = np.clip(lap_bl,0,255).astype(np.uint8)
    
    return final                                    # return final blended image to Main Code

""" function to create Gaussian pyramid for mask """
def create_gPyr_mask(mask,num_layers):
    
    gMask = np.copy(mask).astype(np.float32)
    g_pyr_mask = [gMask]                            # create list of masks
    
    for i in range(num_layers-1):
        gMask = downSampler(conv2(gMask,w))         # pyramid of smaller masks - call downSampler, convolve then downscale
        g_pyr_mask.append(gMask)                    # add layers to list of Gaussian masks
        
    g_pyr_mask.reverse()                            # reverse Gaussian mask list from smaller to bigger to apply to Laplacian list of images
    
    return g_pyr_mask

""" function to upscale img using nearest neighbor interpolation """
def upSampler(img,img_m1):
    
    ogRow = img.shape[0]
    ogCol = img.shape[1]

    newRow = img_m1.shape[0]
    newCol = img_m1.shape[1]
    
    rRatio = newRow / ogRow
    cRatio = newCol / ogCol
    
    fRow = (np.floor((np.arange(0,newRow,1)) / rRatio)).astype(np.int16)
    fCol = (np.floor((np.arange(0,newCol,1)) / cRatio)).astype(np.int16)
    
    uS_img = img[fRow , :]
    uS_img = uS_img[: , fCol]
    
    return uS_img                                   # return uS_img to pyr_up

""" function to downscale img using nearest neighbor interpolation """
def downSampler(img):
    
    ogRow = img.shape[0]
    ogCol = img.shape[1]
    
    newRow = (np.floor(img.shape[0] / 2)).astype(np.int16)
    newCol = (np.floor(img.shape[1] / 2)).astype(np.int16)
    
    rRatio = newRow / ogRow
    cRatio = newCol / ogCol
    
    fRow = (np.floor((np.arange(0,newRow,1)) / rRatio)).astype(np.int16)
    fCol = (np.floor((np.arange(0,newCol,1)) / cRatio)).astype(np.int16)
    
    dS_img = img[fRow , :]
    dS_img = dS_img[: , fCol]
    
    return dS_img                                   # return dS_img to pyr_down

""" function compute Gaussian/Laplacian pyramid """
def ComputePyr(img,num_layers):
    
    global pyr_layers, w                            # minimum layers should be 1 and w is Gaussian kernel
   
    pyr_layers = 1
    shape = img.shape[0]                            # since we're using square images, use either shape is alright
    
    for i in range(num_layers-1):
        if (shape//2 < 5):                          # floor division, cannot be less than kernel size 5x5
            print("Maximum no. of layers computed from the image is %d " %pyr_layers)
            num_layers = pyr_layers                 # num_layers = last increment of pyr_layers in else statement
            break
        else:
            shape = shape // 2                      # as long as shape is floored by two and > 5, increment no. layers in the pyramid
            pyr_layers += 1
    
    """ Gaussian kernel """
    g_kernel = cv2.getGaussianKernel(5,2)           # 1-d Gaussian array, std deviation is 2
    w = g_kernel*(g_kernel.T)                       # create 2-d gaussian filter
    
    """ Gaussian pyramid """
    g = img.copy().astype(np.float32)
    g_pyr_img = [g]                                 # first layer of the Gaussian pyr = original img
    for i in range(num_layers-1):
        g = downSampler(conv2(g,w))                 # pyramid of smaller images - call downSampler, convolve then downscale
        g_pyr_img.append(g)                         # add layers to list of Gaussian images
        
    """ Laplacian pyramid """
    l_pyr_img = [g_pyr_img[num_layers-1]]           # first layer of Laplacian pyr = last layer in Gaussian list of images 
    for i in range(num_layers-1,0,-1):
        G = conv2(upSampler(g_pyr_img[i],g_pyr_img[i-1]),w)         # pyramid of bigger images - call upSampler, upscale then convovle
        l = np.subtract(g_pyr_img[i-1], G)           
        l_pyr_img.append(l)                         # add layers to list of Laplacian images
    
    return g_pyr_img, l_pyr_img                     # return to main code

""" function to align f_img and b_img """
def align(f_img,b_img,row,col):
    
    a_img = np.zeros(b_img.shape)                   # create blank img same size as b_img filled with 0
    a_img[row:(f_img.shape[0]+row) , col:(f_img.shape[1]+col)] = f_img          # aligning f_img with b_img
   
    return a_img.astype(np.unit8)                   # return aligned img to main code