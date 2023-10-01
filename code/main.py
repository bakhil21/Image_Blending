""" import necessary libs """
import numpy as np
import cv2
from P2_Packages import align, ComputePyr, create_gPyr_mask, imgBlending

""" function to create mask from aligned f_img """
def create_mask_fimg(f_imgcp, init_x, init_y, new_x, new_y):
    
    new_mask = np.zeros(f_imgcp.shape).astype(np.float32)
    
    """ OpenCV function to draw rectangle and ellipse """
    if rec_draw == False:
        new_mask = cv2.ellipse(new_mask, ((init_x+(new_x-init_x)//2),(init_y+(new_y-init_y)//2)), (new_x-init_x,new_y-init_y), 0,0,360, (1,1,1), -1)
    else:
        new_mask = cv2.rectangle(new_mask, (init_x,init_y),(new_x,new_y), (1,1,1), -1)
        
    return new_mask                                 # return new_mask to main code



""" function to draw mask on displayed GUI """
def get_mask(event,x,y,f,param):                    # openCV defaut parameters

    global init_x, init_y, new_x, new_y, mouse_pressed  # these vars can be used outside of this function
    
    if event == cv2.EVENT_LBUTTONDOWN:              # left mouse button is pressed
        mouse_pressed = True
        init_x,init_y = x,y                         # current starting x-y mouse position assigned to initial position
        
    elif event == cv2.EVENT_LBUTTONUP:              # left mouse button is released
        mouse_pressed = False
        
        if rec_draw == False:
            cv2.ellipse(f_imgcp, ((init_x+(x-init_x)//2),(init_y+(y-init_y)//2)), (x-init_x,y-init_y), 0,0,360, (0,0,0), 1)
        else:
            cv2.rectangle(f_imgcp, (init_x,init_y), (x,y), (0,0,0), 1)      #show drawn shape
        
        # ending x-y mouse position
        new_x = x;
        new_y = y;


####################---MAIN CODE---####################

""" set up vars for drawing mask """
mouse_pressed = False                               # true if mouse is pressed
rec_draw = True                                     # if True, draw rectangle, press 'e' to toggle to draw ellipse
init_x,init_y = -1,-1                               # initial x-y position of mouse


""" read images """
f_img = cv2.resize(cv2.imread('lc1.jpg'), (940,620))              # foreground img as RGB, optional: grasycale
b_img = cv2.resize(cv2.imread('lc2.jpg'), (940,620))              # background img as RGB, optional: grayscale
 

""" images aligning """
if (b_img.shape[0] > f_img.shape[0]):               # if b_img's no. rows > f_img's no. row
    f_img_align = align(f_img,b_img,50,35)          # row & col can be adjusted manually
    f_img = np.copy(f_img_align)
    f_imgcp = np.copy(f_img_align)                  # copy of aligned_img for mask
else:
    f_imgcp = np.copy(f_img)


""" GUI to draw mask and display mask """
cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Foreground Image', get_mask)      # call get_mask function to draw and define shape

while(1):
    cv2.imshow('Foreground Image',f_imgcp)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('e'):                             # press 'e' to toggle to ellipse
        rec_draw = not rec_draw                     # then draw rectangle set to False and draw ellipse
    elif key == 27:                                 # press 'ECS' to exit window after drawing
        cv2.destroyAllWindows()
        break
    

mask = create_mask_fimg(f_imgcp, init_x, init_y, new_x, new_y)      # create mask from aligned f_img      
                                                                    # new_x, new_y are global in get_mask

""" display f_img, b_img, and mask """
cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',f_img)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask', mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image', b_img)
cv2.waitKey(0)                                      # after dislaying, press any key to close windows
cv2.destroyAllWindows()


""" compute Gaussian/Laplacian pyramid, create Gaussian mask, and blend images """
gPyr_fimg, lPyr_fimg = ComputePyr(f_img,10)
gPyr_bimg, lPyr_bimg = ComputePyr(b_img,10)
gPyr_mask = create_gPyr_mask(mask,len(gPyr_fimg))   
blended_img = imgBlending(lPyr_fimg, lPyr_bimg, gPyr_mask, len(gPyr_fimg)) 


""" testing - write to see images """
#for i in range(len(gPyr_fimg)):
    #cv2.imwrite('gPyr_fimg{i}.png'.format(i=i), gPyr_fimg[i])
    #cv2.imwrite('lPyr_fimg{i}.png'.format(i=i), lPyr_fimg[i])
    #cv2.imwrite('gPyr_bimg{i}.png'.format(i=i), gPyr_bimg[i])
    #cv2.imwrite('lPyr_bimg{i}.png'.format(i=i), lPyr_bimg[i])
    #cv2.imwrite('gPyr_mask{i}.png'.format(i=i), gPyr_mask[i])
    
    
""" display f_imgcp, b_img, mask, and blended_image"""
cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image',f_imgcp)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask',mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image',b_img)
cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blended Image',blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" write result images for report """
cv2.imwrite('f_imgcp.png', f_imgcp)
cv2.imwrite('b_img.png', b_img)
cv2.imwrite('bl_img.png',blended_img)

####################---End of MAIN CODE ---####################