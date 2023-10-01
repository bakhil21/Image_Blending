""" import necessary packages """
import numpy as np

""" padding function """
def pad(image,pad_w,padding):
    
    if padding == "clip/zero-padding":
        
        """ zero-padding algorithm """
        padded_img = np.zeros((image.shape[0] + (pad_w * 2), image.shape[1] + (pad_w * 2)))             # create (pad_w)-zero arrays around the image
        padded_img[int(pad_w):-int(pad_w) , int(pad_w):-int(pad_w)] = image                              # assume padding width is even all around
        
        return padded_img                   # return padded image to conv2
        
        
    if padding == "wrap around":
        
        """ wrap-around padding algorithm """
        prepad_img = image                  # make copy of the passed image from conv2 to do verical stacking

        tl_chunk = image[:pad_w , :pad_w]     # filler extracted from top left of image
        bl_chunk = image[-pad_w : , :pad_w]   #               "       bottom left
        tr_chunk = image[:pad_w , -pad_w:]    #               "       top right
        br_chunk = image[-pad_w: , -pad_w:]   #               "       bottom right

        # image left and right extracter, IMPORTANT: do transpose
        left = np.array([image[:,-1]]).T
        right = np.array([image[:,0]]).T
        
        for i in range(pad_w):
            prepad_img = np.vstack((image[-(1+i),:], prepad_img, image[(0+i),:]))             # stack to top and bottom of prepad_img
            if i < (pad_w - 1):                                                               # offset & stack to left and right extracter
                left = np.hstack((np.array([image[:,-(2+i)]]).T, left))
                right = np.hstack((right, np.array([image[:,(1+i)]]).T))
            else:                                                                             # add filters to extracters
                left_with_chunk = np.vstack((br_chunk, left, tr_chunk))
                right_with_chunk = np.vstack((bl_chunk, right, tl_chunk))

        padded_img = np.hstack((left_with_chunk, prepad_img, right_with_chunk))               # stack to left and right of image
        
        return padded_img                   # return padded image to conv2
        
    
    if padding == "copy edge":
        
        """ copy-edge padding algorithm """
        prepad_img = image
        
        tl_chunk = np.full((pad_w,pad_w),image[0][0])                        
        bl_chunk = np.full((pad_w,pad_w),image[-1][0])                      # fillers extracted from edges of image - arrays filled with values
        tr_chunk = np.full((pad_w,pad_w),image[0][-1])
        br_chunk = np.full((pad_w,pad_w),image[-1][-1])

        left = np.array([image[:,0]]).T
        right = np.array([image[:,-1]]).T
    
        for i in range(pad_w):
            prepad_img = np.vstack((image[0,:], prepad_img, image[-1,:]))                     # stack to top and bottom of prepad_img
            if i < (pad_w-1):                                                                 # offset & stack to left and right extracter
                left = np.hstack((np.array([image[:,0]]).T, left))
                right = np.hstack((right, np.array([image[:,-1]]).T))
            else:                                                                             # add filters to extracters
                left_with_chunk = np.vstack((tl_chunk, left, bl_chunk))
                right_with_chunk = np.vstack((tr_chunk, right, br_chunk))

        padded_img = np.hstack((left_with_chunk,prepad_img,right_with_chunk))                 # stack to left and right of prepad_image
        
        return padded_img                   # return padded image to conv2

   
    if padding == "reflect across edge":
        
        """ reflect-across-the-edge padding algorithm """
        prepad_img = image
        
        for i in range(pad_w):
            prepad_img = np.vstack((image[(1+i),:], prepad_img, image[-(2+i),:]))              # stack to top and bottom of prepad_img

        padded_img = prepad_img

        for i in range(pad_w):
            padded_img = np.hstack((np.array([prepad_img[:,(1+i)]]).T, padded_img, np.array([prepad_img[:,-(2+i)]]).T))     # stack to left and right of prepad_img
        
        return padded_img                   # return padded image to conv2

""" function performing comvolution """
def conv2(f,w):
    
    
    w_width, w_height = w.shape             # get kernel's width and length
    
    pad_w = 2                               # get input for padding width
    stride = 1                              # get input for the number of stride
    padding = 'reflect across edge'         # using "reflect" padding type
    
    if (len(f.shape) < 3):                  # check if passed image argument is grayscale
         
        """ convolution algorithm for grayscale """  
        f = pad(f,pad_w,padding)            # call pad function to do padding, pass grayscale image (f), padding width (pad_w), and padding type
        padding_width, padding_height = f.shape                             # get padding width and height
        # paddled image
        new_width = (padding_width - w_width) // stride + 1                   
        new_height = (padding_height - w_height) // stride + 1
        new_f = np.zeros((new_width,new_height)).astype(np.float32)
            
        for x in range(0,new_width):
            for y in range(0,new_height):
                new_f[x][y] = np.sum(f[x * stride:x * stride + w_width, y * stride:y * stride + w_height] * w).astype(np.float32)
                
        return new_f                        # return filtered image back to main function
        
   
    if (len(f.shape) == 3):                 # check if passed image argument is RGB
        
        fB = f[:,:,0]
        fG = f[:,:,1]                       # split RGB image into sub-R/G/and B image to perform 2d convolution
        fR = f[:,:,2]
         
        """ convolution algorithm for RGB """
        fB = pad(fB,pad_w,padding)
        fG = pad(fG,pad_w,padding)                         # do padding for each sub-image, call pad function to do padding
        fR = pad(fR,pad_w,padding)
        # get padding width and height. fG or fR.shape also work since they have same dimensions
        padding_width, padding_height = fB.shape  
        # paddled image sub-images
        new_width = (padding_width - w_width) // stride + 1
        new_height = (padding_height - w_height) // stride + 1
        new_fB = np.zeros((new_width,new_height)).astype(np.float32)
        new_fG = np.zeros((new_width,new_height)).astype(np.float32)
        new_fR = np.zeros((new_width,new_height)).astype(np.float32)
            
        for x in range(0,new_width):
            for y in range(0,new_height):
                new_fB[x][y] = np.sum(fB[x * stride:x * stride + w_width, y * stride:y * stride + w_height] * w).astype(np.float32)
                new_fG[x][y] = np.sum(fG[x * stride:x * stride + w_width, y * stride:y * stride + w_height] * w).astype(np.float32)
                new_fR[x][y] = np.sum(fR[x * stride:x * stride + w_width, y * stride:y * stride + w_height] * w).astype(np.float32)
        
        new_rgb = np.dstack((new_fB,new_fG,new_fR))         # depth stack three channels B|G|R back into 1 RGB image
        
        return new_rgb                      # return filtered image back to main function