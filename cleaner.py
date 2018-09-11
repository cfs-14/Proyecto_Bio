#import the packages
import numpy as np
import cv2
import itertools as it
import math
import sys
##
    ##
    
    ##(!) Not the best eyes are getting picked, esp horizontally.
        ##Fixed. min_Dist_L & R were not getting updated when new smallest found in num_eyes > 2.
        
    ##(!)If possible, fix the small inconsistency using point_rotation_calc to get to the eyes. 
        ##Consider the compensation used in the mat that accomodates the rotated image; this is probably the reason for the discrepancy.
        ##(!) Tried changing point_rotation_calc, using round but didn't work.
    ##Do the eye handling    
        
        ##ctrl home goes to top of page
        ##higlighting whole line and ctrl+shift up or down swaps line.
        ##(!) when calling transform_image(), send the original image, but update the left_eye and right_eye so that we can crop from the
            ##original image and get a full image w/o showing missing cropped sides. See (C) #Check Works now. 
        ## Make another transform_image, that keeps the ratio of the head vertically and horizontally, not just the face, so as to preserve the whole head.
        
        ##(Opt.?) perhaps callibrate the face rotation? Works fine, but overshoots, perhaps inherent in this face detection.
        ##Handle no face detections
        ##Handle image name is wrong. or no images.
        
        
        ##Add the crop checking: Check to see if it's cropped or not esp. for right_eye, so it's good.
        ##The bx, by, wx, hy needs to be updated when the face is cropped. 
        
        ##Try altering the threshold of the image first before changing the scale_factor. Check. Does help.
        ##Must now rotate the face to find the best angle. 
        ##If I wanted to overdo it, I would rotate the pic and keep track of all the faces, and have their best confidences ||> to reduce work, keep the best confidence one.
        ##To save resources, I should just get the best face from the rotations of the entire image.
            ##Then expand the area of the face and rotate this face ROI to find the best angle. We can also do threshold stuff.
                ##I realize this will help alot in filtering false positives for the eyes. (!) Test on walter_angle.jpg.
    ##Process the sequence of pics.
    ##Take the sequence of pics.
    
    ##Do the scale: check.    
        ##decide on interpolation method
    ##Do the confidence formula check.
        ##for x y check.
        ## for size of face. check

    ###Seem to have fixed the crop for good. Check.
    
    #Why horizontal dots are not aligned? Check.
        #Because dist_o isn't precise. As in rotated_eye_L[0]+dist_o != rotated_eye_R[0], yet will still work fine.
        ##The real reason horizontal dots are not aligned for offset = (.3, .3) is because 1/offset is not an even hside_o.
    ##.3 different from gender class, prob because we give an roi to transform_image(). Check.
        ##Yes, it is because we gave an roi, image is not large enough to accomodate the offset percentage provided. 
    ##(!) See (B). Solved: cropping works, just when original image is too short on either side to accomodate percentage, we use the max we can afford.
    ##(!) See (N). Solved: We have to send the original image to get as much of the original image as we can insted of ROI of the face.
    ##Do scaling
    ##Do the sequence of images. 
    
    ##Clean more pussy
    ##This is a copy of wip.py that I need to clean up.
    ##I should delete/strip all the functions and replace them w/ the ones in drawing.py, since they work.
    ##I conjecture that I will keep the code outside the functions.
    ##I will need to add the cropping and scaling.
        ##Cropping works. (!)(N) But we have to keep the entire picture so as to not cut off a large portion. Ideally, keep twice the size of the original 
            ##bounding box, but use what? face can be close to camera or far. so either math.min(x+x*.5  or img.shape[1])
        
        
        ##(!)Do the scaling.
        ##_Perhaps make the scaling have to do w/ it.
        ##
        # #(!)(Done) after draw a circle every i*vside_o to see if it matches up.
    ##cropping at offset_pct(.1, .1) gives a gray margin or right side. Try merging?
    
    ##(!)(A)<- ctrl+f 
    ##Find the right proportions to get the face.
    ##use the same jpg face in the other example to see if it matches up.
    ##
    
    ##I will need to test it of course.
    ##
    ##Get an idea of:
    ##Barcode control, and get info && from database.
    ##GPS control.
    ##
    
##

#(!)Caution, this relies on the fact that there will be no upside down faces.
#Caution, this relies also that upright faces will have a higher confidence score
#_than any upside down face detected, including the same face upside down.

#If time & processing was not an obstacle, then, upon detecting a face w/in a threshold of    
    #_ best_image's confidence, then we sweep rotate the area of this candidate face
    #_ in degrees of 1, to see if we can get improved confidence looking for one that beats the current
    #_ best_image. This will also be normalizing our image for us somewhat, as a side effect.
    
#(!) one trick I can use for the eyes, is getting the best angle for every face in a list, along w/ the bounding box, and limit the eye search to the upper half of the box
#_since it's the corners of the mouth that get confused.
#(!) (Test) run this with a blank image to see how it fails.
#(!)(C) Create a function to show face in window.
#(!)(C) Alter CropFace or really set up the default eyes, or eyes backup plan.

##################################################################
##################################################################

def distance(p1,p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)

################################################################
##################

##perhaps have an extra precaution by giving default values to the eyes, but not so the distance is 0.
##def transform_image(image, angle, eye_L, eye_R, offset_pct=(0.5,0.5), dest_sz=(300.0, 300.0)): testing: (.3, .25)
def transform_image(image, eye_L, eye_R, offset_pct=(0.15,0.15), dest_sz=(300, 300)): #offset_pct must be b/w:  [ 0.0 < (0.1 <= offset_pct <= 0.3 ) < 0.5] to make sense & work correctly
    """
    """
    
##Rotate
    # get the direction
    eye_direction = (eye_R[0] - eye_L[0], eye_R[1] - eye_L[1])
    # calc rotation angle in radians
    angle = math.atan2(float(eye_direction[1]),float(eye_direction[0]))
    #rotate_image works with degrees so we have to convert them to degrees
    angle = math.degrees(angle)
    #we rotate around the left eye (center, angle, scale)
    print("angle:")
    print(angle)        
    
    alt_image = rotate_image(image, angle, eye_L)
    cp_image = rotate_image(image, angle, eye_L)
    
    show_image_destroy(alt_image, 'After Rotating')
    
##Cropping    
    #We create a test image to make sure we are calculting cropping & scaling correctly    
    test_image = np.zeros([int(dest_sz[0]), int(dest_sz[1]), 3], dtype=np.uint8) #height, width, channels.        
        
    #o for original
    ##make dist_o not zero.
    dist_o = distance(eye_L, eye_R)
    offset_h, offset_v = offset_pct
    
    rotated_eye_L = point_rotation_calc(image, angle, eye_L, eye_L)

    offset_correct = True   
    
    #(!) Should prob initialize better vv
    hside_new = vside_new = dist_new = 0
    hside_o = vside_o = 0
    sfx = sfy = 1.0
    
    #while we do not have the correct offset percentages.
    while True:
    #do while (offset_correct) offset_correct = true; if(hside_o < hside_new): offset_correct = false; adjust offset_h; if(vside_o < vside_new): offset_correct = false; adjust offset_v; 
        #Now we get the length of lines in the scaled & cropped image
        offset_correct = True
        print("Correct?:")
        print(offset_correct)
        
        hside_new = offset_h*dest_sz[0]
        vside_new = offset_v*dest_sz[1]
        dist_new = dest_sz[0]- 2*hside_new
        
        #The ratio of the dist. b/w the eyes in dest. to the height above the eyes in dest.
        ratioHTV = vside_new/dist_new
        
        # # #we draw hside_new L
        cv2.line(test_image, (0, int(vside_new)), (int(hside_new), int(vside_new)),
            (255, 0, 0), 2)
        #we draw hside_new R
        cv2.line(test_image, (int(dest_sz[0]), int(vside_new)), (int(dest_sz[0]-hside_new), int(vside_new)),
            (255, 0, 0), 2)
        
        #we draw vside_new L
        cv2.line(test_image, (int(hside_new), 0), (int(hside_new), int(vside_new)),
            (0, 255, 0), 2)        
        #we draw vside_new R
        cv2.line(test_image, (int(dest_sz[0]-hside_new), 0), (int(dest_sz[0]-hside_new), int(vside_new)),
            (0, 255, 0), 2)
        
        #we draw the dist_new
        cv2.line(test_image, (int(hside_new), int(vside_new)), (int(hside_new+dist_new), int(vside_new)),
            (0, 0, 255), 2)
        
        show_image_destroy(test_image, 'Test')
        
        
        #The length of imp. lines in original image to know where to crop.
        #we get the x & y scaling factors to use sfy right now, but to use them to scale later.
        vside_o = dist_o*ratioHTV
        
        sfx = dist_new/dist_o
        sfy = vside_new / vside_o
        
        
        hside_o = (1/sfx)*hside_new        
       
       ##(!)(?) hside_o = math.floor(hside_o)? or when I calc/create hside_o 
        #we check if we can accomodate these percentages.
        if rotated_eye_L[0] < int(hside_o):
            #we must recalculate offset_h to be the max it can accomodate.
            #(!)Log that the offset pct width was too large for the dest_sz?
            offset_correct = False
            #should always be < 1.
            offset_h = rotated_eye_L[0]/((2*rotated_eye_L[0])+dist_o)
        if rotated_eye_L[1] < int(vside_o):
            #(!)Log that the offset pct width was too large for the dest_sz?
            offset_correct = False
            #the offset percentage of the height will become the percent that the y segment above the eye already is to the height of the image.
            offset_h = rotated_eye_L[1]/alt_image.shape[0]
            
        #should be twice max, since it doesn't seem to reduce less than the dimensions of alt_image.
        if(offset_correct):
            break        
    ##endWhile

    ## draw on the new image
    
    ##commented out
    #we draw hside_o L    
    # cv2.line(alt_image, (int(rotated_eye_L[0]-hside_o), int(rotated_eye_L[1])), (int(rotated_eye_L[0]), int(rotated_eye_L[1])),
        # (255, 0, 0), 2)
    # #we draw hside_o R
    # cv2.line(alt_image, (int(rotated_eye_L[0]+dist_o), int(rotated_eye_L[1])), (int(rotated_eye_L[0]+dist_o+hside_o), int(rotated_eye_L[1])),
        # (255, 0, 0), 2)
    
    # #we draw vside_o L
    # cv2.line(alt_image, (int(rotated_eye_L[0]), int(rotated_eye_L[1])), (int(rotated_eye_L[0]), int(rotated_eye_L[1]-vside_o)),
        # (0, 255, 0), 2)        
    # #we draw vside_o R
    # cv2.line(alt_image, (int(rotated_eye_L[0]+dist_o), int(rotated_eye_L[1])), (int(rotated_eye_L[0]+dist_o), int(rotated_eye_L[1]-vside_o)),
        # (0, 255, 0), 2)
    
    # #we draw the dist_o
    # cv2.line(alt_image, (int(rotated_eye_L[0]), int(rotated_eye_L[1])), (int(rotated_eye_L[0]+dist_o), int(rotated_eye_L[1])),
        # (0, 0, 255), 2)
    
    # show_image_destroy(alt_image, 'Test Alt')
    
    ##
    ##crop it
    
    #show bottom right corner of crop.
    cv2.circle(alt_image, (int(math.floor(int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)))), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))), 3, 
     (0, 0, 0), -1)    
    cv2.circle(alt_image, (int(math.floor(int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)))), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))), 2, 
     (0, 255, 0), -1)      
    
    show_image_destroy(alt_image, 'Bottom Right')
    
    
    
    ##commented out the dots.
    
    # count2 = 0
    # #Y
    # while count2 <= 1/offset_v:
        # cv2.circle(alt_image, (int(rotated_eye_L[0]), int((rotated_eye_L[1]-vside_o)+count2*vside_o)), 3, #
            # (0, 0, 0), -1)
        # cv2.circle(alt_image, (int(rotated_eye_L[0]), int((rotated_eye_L[1]-vside_o)+count2*vside_o)), 2, #
            # (0, 255, 255), -1)
            
        
        # count2+=1
    
    # count2 = 0
    
    #The problem is that the rotated left eye is not correct.
    
    ##draw random reference dots.

    #  (rotated_eye_L[0]-int(hside_o), rotated_eye_L[1])
    # cv2.circle(alt_image, (int(rotated_eye_L[0]-hside_o), int(rotated_eye_L[1])), 3,
        # (0, 0, 0), -1)
    # cv2.circle(alt_image, rotated_eye_L, 3,
        # (0, 0, 0), -1)
    # # cv2.circle(alt_image, (int(hside_o), 0), 5,
        # # (0, 250, 250), -1)
        
    # show_image_destroy(alt_image, 'vert')
    ##dots     
    
    ##X
    # while count2 <= 1/offset_h: 
        # cv2.circle(alt_image, (int((rotated_eye_L[0]-hside_o)+count2*hside_o), int(rotated_eye_L[1])), 3, #
            # (0, 0, 0), -1)
        # cv2.circle(alt_image, (int((rotated_eye_L[0]-hside_o)+count2*hside_o), int(rotated_eye_L[1])), 2, #
            # (255, 255, 0), -1)
            
        # cv2.circle(alt_image, (int((rotated_eye_L[0]+dist_o+hside_o)-count2*hside_o), int(rotated_eye_L[1])), 3, #
            # (0, 0, 0), -1)
        # cv2.circle(alt_image, (int((rotated_eye_L[0]+dist_o+hside_o)-count2*hside_o), int(rotated_eye_L[1])), 2, #
            # (255, 0, 255), -1)
            
        # count2+=1
    
    # cv2.circle(alt_image, (rotated_eye_L[0]+int(dist_o), rotated_eye_L[1]), 2,
        # (0, 250, 250), -1)
    # cv2.circle(alt_image, (rotated_eye_L[0]-int(hside_o), rotated_eye_L[1]), 2,
        # (0, 250, 250), -1)
    # show_image_destroy(alt_image, 'horizontal')
    
    ##end horz dots
    # show_image(alt_image, 'Before Cropping')
    
    ##commented out.
    # print("full width {} & end crop x: {}".format(alt_image.shape[1], int(rotated_eye_L[0]+dist_o+hside_o)))
    # print("Dimensions of alt_image: width{}, height{}".format(alt_image.shape[1], alt_image.shape[0]))
    # print("\tStartx: {}\n\tEndx: {}\n\tStarty: {}\n\tEndy: {}".format(int(rotated_eye_L[0]-hside_o), 
            #int(rotated_eye_L[0]+dist_o+hside_o), int(rotated_eye_L[1]-vside_o), int(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o))))
        
    cv2.circle(alt_image, (int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))), 2, 
     (0, 255, 0), -1)
     
    # #draw rectangle around cropped area.
    # cv2.rectangle(alt_image, (0, 0), (int(int(rotated_eye_L[0]+dist_o+hside_o)-1), int(int(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o))-1)), #(int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))),
        # (255, 0, 255), 1)
    
    # ##Here Now
    # ##So it seems that it is cropping correctly, it's just that the window is a little bigger than we want. 
    # show_image_destroy(alt_image, 'Crop Area')
    
    #print("Shape: {}".format(alt_image.shape))
    
    ##commented out.
    # # print("Eye.x: {}".format(rotated_eye_L[0]))
    # # print("hside_o: {}".format(hside_o))
    
    # # print("_")
    # # print("\tStartX:{}".format(rotated_eye_L[0]-hside_o))
    # # print("\tToX:{}".format(int(math.ceil(rotated_eye_L[0]+dist_o+hside_o))))
    # # print("\tStartY:{}".format(int(math.floor(rotated_eye_L[1]-vside_o))))
    # # print("\tToY:{}".format(int(math.ceil(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))))
    
    #precautionary since hside_o, for example, a float, can be greater than rotated_eye_L[0] thus resulting in a negative index.
    # cropStX = 0 if int(math.floor(rotated_eye_L[0]-hside_o)) < 0 else int(math.floor(rotated_eye_L[0]-hside_o))
    # cropEnX = alt_image.shape[0] if int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)) > alt_image.shape[0] else int(math.ceil(rotated_eye_L[0]+dist_o+hside_o))
    # cropStY = 0 if int(math.floor(rotated_eye_L[1]-vside_o)) < 0 else int(math.floor(rotated_eye_L[1]-vside_o))
    # cropEnY = alt_image.shape[1] if int(math.ceil(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o))) > alt_image.shape[1] else int(math.ceil(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))
    
    ##Optimize somehow:
    cropStX = int(math.floor(rotated_eye_L[0]-hside_o))       if int(math.floor(rotated_eye_L[0]-hside_o)) >= 0 else 0
    cropEnX = int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)) if int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)) <= alt_image.shape[1] else alt_image.shape[1]
    cropStY = int(math.floor(rotated_eye_L[1]-vside_o))       if int(math.floor(rotated_eye_L[1]-vside_o)) >= 0 else 0
    cropEnY = int(math.ceil(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o))) if int(math.ceil(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o))) <= alt_image.shape[0] else alt_image.shape[0]
    
    
    
    
    cv2.rectangle(alt_image, (cropStX, cropStY), (cropEnX, cropEnY), #(int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))),
        (255, 255, 0), 1)
    
    ##Here Now
    ##So it seems that it is cropping correctly, it's just that the window is a little bigger than we want. 
    show_image_destroy(alt_image, 'Crop Area')
    
    
    #alt_image = alt_image[int(math.floor(rotated_eye_L[1]-vside_o)): int(math.ceil(rotated_eye_L[1]+(10*(1-offset_pct[1])*vside_o))), int(math.floor(rotated_eye_L[0]-hside_o)):int(math.ceil(rotated_eye_L[0]+dist_o+hside_o))]
    ##most recent working one:alt_image = alt_image[int(math.floor(rotated_eye_L[1]-vside_o)): int(math.ceil(rotated_eye_L[1]+(((1/offset_pct[1])-1)*vside_o))), int(math.floor(rotated_eye_L[0]-hside_o)):int(math.ceil(rotated_eye_L[0]+dist_o+hside_o))]
    ##(!)(!)(B) print the values used here in the croppping and compare them to the shape. Also, substitute literal nubmers to see if it does the same.
    #alt_image = alt_image[int(rotated_eye_L[1]-vside_o): int(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)), int(rotated_eye_L[0]-hside_o):int(rotated_eye_L[0]+dist_o+hside_o)]
    #When depended on offset_pct: [int(rotated_eye_L[1]-vside_o): int(rotated_eye_L[1]+(((1/offset_pct[1])-1)*vside_o)), int(rotated_eye_L[0]-hside_o):int(rotated_eye_L[0]+dist_o+hside_o)]
    ##Before, had not ceil or floor, but works better w/ ceil & floor.
    alt_image = alt_image[cropStY:cropEnY , cropStX:cropEnX]
    cp_image = cp_image[cropStY:cropEnY , cropStX:cropEnX]
    

    #del. used to debug crop
    # print("$$$$Shape: {}".format(alt_image.shape))
    # cv2.circle(alt_image, (int(math.ceil(rotated_eye_L[0]+dist_o+hside_o)), int(math.floor(rotated_eye_L[1]+(((1/offset_v)-1)*vside_o)))), 2, 
     # (0, 255, 0), -1)
     
    cv2.circle(alt_image, (alt_image.shape[1], alt_image.shape[0]), 2, 
     (255, 255, 0), -1)
    
    ##commented out
    # print('\n')
    # print('Offsets (hz, vt):')
    # print(offset_pct)
    # print('Offsets inverse')
    # print((1-offset_h))
    
    show_image_destroy(alt_image, 'After Cropping')
    
    #find the point to where the (center) original center point will rotate to. 
    # qx = (math.cos(radians_angle)*(center[0] - pivot[0]) - math.sin(radians_angle)*(center[1] - pivot[1])) + pivot[0]
    # qy = (math.sin(radians_angle)*(center[0] - pivot[0]) + math.cos(radians_angle)*(center[1] - pivot[1])) + pivot[1]     
    
    ##now we get the x & y scaling factors to scale it
    
    ##draw it on a new empty mat the size of dest_sz
##Scale
    
    interpol = cv2.INTER_LINEAR #for zooming in
    if sfx > 1.0 and sfy > 1.0:
        interpol = cv2.INTER_AREA #for shrinking (zooming out)        
    #(src, dsize[dst fx, fy, interpolation)
    
    #alt_image = cv2.resize(alt_image, dest_sz, fx=sfx, fy=sfy, interpolation=interpol)
    
    # im1 = cv2.resize(cp_image, None, test_image, sfx, sfy, interpolation=interpol)
    
    # im2 = cv2.resize(cp_image, dest_sz, interpolation=interpol)
    
    # im3 = cv2.resize(cp_image, dest_sz, fx=sfx, fy=sfy, interpolation=interpol)
    
    # show_image(im1, "One")
    # show_image(im2, "Two")
    # show_image(im3, "Three")
    # ##for some reason, two looks better...?
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #We should definitely get the ratio/proportions so that we also get the chin, and maybe the top of the head.

    #we resize:
    #(Opt.) we can check if the image is shrinking or growing to use a more appropriate interpolation.   
    
    return alt_image

####################################
def rotate_image(mat, angle, pivot=None): #Info: positive angle in degrees rotates counter-clockwise.
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    radians_angle = math.radians(-1*angle) ##(!)(?) why -1* again?
    
    #if no pivot given, make pivot the center of image.
    if pivot is None:
        pivot = (mat.shape[1]/2, mat.shape[0]/2)
    
    width, height = mat.shape[1], mat.shape[0]

    center = (width/2, height/2)
    #image_center = pivot #(mat.shape[1]/2, mat.shape[0]/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape    

    image_copy = mat.copy()
    
    # #draw the pivot point.    
    # cv2.circle(image_copy, pivot, 2, 
     # (0, 255, 100), -1)
    
    # show_image(image_copy, 'Pivot') #_destroy

    #commented out
    # print('Pivot')
    # print(pivot)
    # print('Angulo')
    # print(angle)
    
    # M = np.float32([[1,0,100],[0,1,50]])
    # dst = cv.warpAffine(img,M,(cols,rows))
    
    #(Info:) matrix = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    #blank_rmat = cv2.getRotationMatrix2D((width/2, height/2), 0, 1.)
    ##
    # print('Orig Rotation Matrix')
    # print(blank_rmat)
    
    ##(ch) pivot -> center
    rotation_mat = cv2.getRotationMatrix2D(pivot, angle, 1.)#(image_copy.shape[1]/2, image_copy.shape[0]/2), angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    
    abs_cos = abs(rotation_mat[0,0]) # #math.cos(angle - 45)
    abs_sin = abs(rotation_mat[0,1]) #  math.sin(angle - 45))
    
    ##
    # print('Rotation Matrix')
    # print(rotation_mat)        
    
    # find the new width and height bounds
    bound_w = int( math.ceil(height * abs_sin + width * abs_cos+2)) # hypotenuse* math.cos(angle - math.atan(height/width)
    bound_h = int( math.ceil(height * abs_cos + width * abs_sin+2)) #  hypotenuse* math.cos(angle - math.atan(width/height)    
    
    new_center = (bound_w/2, bound_h/2)
    
    #find the point to where the (center) original center point will rotate to. 
    qx = (math.cos(radians_angle)*(center[0] - pivot[0]) - math.sin(radians_angle)*(center[1] - pivot[1])) + pivot[0]
    qy = (math.sin(radians_angle)*(center[0] - pivot[0]) + math.cos(radians_angle)*(center[1] - pivot[1])) + pivot[1]      
    
    
    # #draw the center after rotation
    # cv2.circle(image_copy, (int(qx), int(qy)), 2, 
     # (0, 0, 0), -1)
     
    # #draw the new center w/ the new bounds to see where we are translating to.
    # cv2.circle(image_copy, new_center, 2, 
     # (250, 250, 50), -1)
     
    # show_image(image_copy, "centerpoint rotated")
    
    #we add the translation that we are supposed to have calculated from 
    rotation_mat[0, 2] += new_center[0] - qx
    rotation_mat[1, 2] += new_center[1] - qy 
    
    # print('New Rotation Matrix')
    # print(rotation_mat)
    
    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image_copy, rotation_mat, (bound_w, bound_h)) ## mat (bound_w, bound_h) is the dest. mat. when we crop we use the original size before the rotation. We delete from the sides.
    
    # #draw previous position rect 
    ## # cv2.rectangle(rotated_image, (0,0),(width,height),(150, 150, 150), 2)###
    
    # #draw pivot to see if still the same
    # cv2.circle(rotated_image, pivot, 5, 
     # (0, 0, 0), -1)
     
    # show_image(rotated_image, 'After rotation')        
    
    return rotated_image
    
####################################
def point_rotation_calc(image, angle, point, pivot=None): 
    """
    Calculates point to where it would have moved after using rotate_image on it.
        The idea is basically to get the original center, rotate it about pivot.
        Find the new center of the dest. 
        Apply to point the translation we would apply the the rotated original center to ge to the new center of dest.
    """
    radians_angle = math.radians(-1*angle) ##(!)(?) why -1* again?
    
    #if no pivot given, make pivot the center of image.
    if pivot is None:
        pivot = (image.shape[1]/2, image.shape[0]/2)
    
    width, height = image.shape[1], image.shape[0]

    center = (width/2, height/2)

    image_copy = image.copy()
    
    # #draw the pivot point.    
    # cv2.circle(image_copy, pivot, 2, 
     # (0, 255, 100), -1)
    
    # show_image(image_copy, 'Pivot') #_destroy

    ##commented out 
    ## print('Pivot')
    ## print(pivot)
    ## print('Angulo')
    ## print(angle)
    
    # M = np.float32([[1,0,100],[0,1,50]])
    # dst = cv.warpAffine(img,M,(cols,rows))
    
    #(Info:) matrix = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    #blank_rmat = cv2.getRotationMatrix2D((width/2, height/2), 0, 1.)
    ##
    # print('Orig Rotation Matrix')
    # print(blank_rmat)
    
    ##(ch) pivot -> center
    rotation_mat = cv2.getRotationMatrix2D(pivot, angle, 1.)#(image_copy.shape[1]/2, image_copy.shape[0]/2), angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    
    abs_cos = abs(rotation_mat[0,0]) # #math.cos(angle - 45)
    abs_sin = abs(rotation_mat[0,1]) #  math.sin(angle - 45))
    
    ##
    # print('Rotation Matrix')
    # print(rotation_mat)        
    
    # find the new width and height bounds
    bound_w = int( math.floor(height * abs_sin + width * abs_cos+2)) # hypotenuse* math.cos(angle - math.atan(height/width)
    bound_h = int( math.floor(height * abs_cos + width * abs_sin+2)) #  hypotenuse* math.cos(angle - math.atan(width/height)    
    
    new_center = (bound_w/2, bound_h/2)
    
    #find the point to where the (center) original center point will rotate to. 
    qx = (math.cos(radians_angle)*(center[0] - pivot[0]) - math.sin(radians_angle)*(center[1] - pivot[1])) + pivot[0]
    qy = (math.sin(radians_angle)*(center[0] - pivot[0]) + math.cos(radians_angle)*(center[1] - pivot[1])) + pivot[1]      
    
    
    # #draw the center after rotation
    # cv2.circle(image_copy, (int(qx), int(qy)), 2, 
     # (0, 0, 0), -1)
     
    # #draw the new center w/ the new bounds to see where we are translating to.
    # cv2.circle(image_copy, new_center, 2, 
     # (250, 250, 50), -1)
     
    # show_image(image_copy, "centerpoint rotated")
    
    #we add the translation that we are supposed to have calculated from 
    # point[0] += new_center[0] - qx
    # point[1] += new_center[1] - qy 
    
    
    new_point = (int(point[0] + new_center[0] - qx), int(point[1] + new_center[1] - qy))
    return new_point
    
####################################
def rotate_point(point, angle, pivot):
    """
    Rotates a point in degrees
    """
    ##first we translate the point to the origin
    #We reverse the angle
    angle*=-1
    #we get args
    cx, cy = pivot
    px, py = point
    
    cs = math.cos(angle)
    sn = math.sin(angle)
    
    #
    px-=cx
    py-=cy
    
    # we move the point from relation to center, to origin
    newx = px*cs - py*sn
    newy = px*sn + py*cs    
    
    px = newx+cx
    py = newy+cy
    
    return (int(px), int(py))
####################################
def get_centerpoint(mat, angle):
    """
    Calculates the center of the image based on an angle rotation
    """
    
    height, width = mat.shape[:2]
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    
    return (int(rotation_mat[0, 2]), int(rotation_mat[1, 2]))#circle argument expects ints
    
####################################
def size_rotated_image(mat, angle, pivot=None): #Info: positive angle in degrees rotates counter-clockwise.
    """
    Finds the dimensions of an image rotated by angle.
    """
    radians_angle = math.radians(-1*angle) ##(!)(?) why -1* again?
    
    #if no pivot given, make pivot the center of image.
    if pivot is None:
        pivot = (mat.shape[1]/2, mat.shape[0]/2)
    
    width, height = mat.shape[1], mat.shape[0]

    center = (width/2, height/2)
    #image_center = pivot #(mat.shape[1]/2, mat.shape[0]/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape    

    image_copy = mat.copy()
    
    # #draw the pivot point.    
    # cv2.circle(image_copy, pivot, 2, 
     # (0, 255, 100), -1)
    
    # show_image(image_copy, 'Pivot') #_destroy

    #commented out
    # print('Pivot')
    # print(pivot)
    # print('Angulo')
    # print(angle)
    
    # M = np.float32([[1,0,100],[0,1,50]])
    # dst = cv.warpAffine(img,M,(cols,rows))
    
    #(Info:) matrix = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    #blank_rmat = cv2.getRotationMatrix2D((width/2, height/2), 0, 1.)
    ##
    # print('Orig Rotation Matrix')
    # print(blank_rmat)
    
    ##(ch) pivot -> center
    rotation_mat = cv2.getRotationMatrix2D(pivot, angle, 1.)#(image_copy.shape[1]/2, image_copy.shape[0]/2), angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    
    abs_cos = abs(rotation_mat[0,0]) # #math.cos(angle - 45)
    abs_sin = abs(rotation_mat[0,1]) #  math.sin(angle - 45))     
    
    # find the new width and height bounds
    bound_w = int( math.ceil(height * abs_sin + width * abs_cos+2)) # hypotenuse* math.cos(angle - math.atan(height/width)
    bound_h = int( math.ceil(height * abs_cos + width * abs_sin+2)) #  hypotenuse* math.cos(angle - math.atan(width/height)    
    
    #new_center = (bound_w/2, bound_h/2)
    return (bound_w, bound_h)
#########################    
def get_diff_translation(point1, point2):
    return (point2[0]-point1[0], point2[1]-point1[1])

def translate_2D_point(point1, translation):
    return (point1[0]+translation[0], point1[1]+translation[1])    
#########################
    
def show_image(image, name='Formiga', resize=False):

    cv2.imshow(name, image)

    if resize:
        imWidth = int(image.shape[0]*0.9)#0.75)
        imHeight = int(image.shape[1]*0.9)#0.8)

        cv2.resizeWindow(name, imHeight, imWidth)

    cv2.waitKey(0)
    
def show_image_destroy(image, name='Formiga', resize=False):

    cv2.imshow(name, image)

    
    if resize:
        imWidth = int(image.shape[0]*0.9)#0.75)
        imHeight = int(image.shape[1]*0.9)#0.8)

        cv2.resizeWindow(name, imHeight, imWidth)

    cv2.waitKey(0) 

    cv2.destroyWindow(name)
    #cv2.destroyAllWindows()

#############################
#can detect upside down faces.
#load our serialized model from disk
nnet = cv2.dnn.readNetFromCaffe('C:/Users/gabav/Desktop/CarlosDirectory/Python_Stuff/opencv/sources/samples/dnn/face_detector/deploy.prototxt', 'C:/Users/gabav/Desktop/CarlosDirectory/Python_Stuff/res10_300x300_ssd_iter_140000.caffemodel')
#'deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#, )
eye_detector = cv2.CascadeClassifier('C:/Users/gabav/Desktop/CarlosDirectory/Python_Stuff/opencv/build/etc/haarcascades/haarcascade_eye.xml')

# Load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
##imageOne = cv2.imread('C:\Users\gabav\Desktop\CarlosDirectory\CFS_Images\line_ppl.jpg')
image = cv2.imread('C:/Users/gabav/Desktop/CarlosDirectory/CFS_Images/Arnie_MultiEyed.jpg') #No_Eyes_Arnie.jpg') #Arnie.jpg') #walter_twizzler.jpg') #Arnie_Triclops.jpg') #2ndPic.jpg') #walter_angle.jpg') #walter_twizzler.jpg') ##walter.jpg') # #Arnie.jpg') # #   # # #line_ppl.jpg') #Arnie.jpg') # # #grp_ppl_2.jpg')
# just get the height and width of the image.
#(Optimization) Make the below a single unit/class.
best_face = np.zeros(1)
best_angle = 0
best_confidence = 0.0
bestX, bestY, bestEndX , bestEndY = 0, 0, 0, 0
#When we use sec_best_face, we check if we even found a second best face.
sec_best_face = np.zeros(1)
sec_best_angle = 0
sec_best_confidence = 0.0

foundNew = False

for angle in it.chain(range(0, -40, -10), range(10, 40, 10)): #range(0, 1):#(10, 11) (0, 1):#(0, 330, 30):#30, 210, 90): #(0, 330, 30): #it.chain(range(0, -40, -10), range(10, 40, 10)):  #(!) Good for testing one eye found in line_ppl: angles (30, 31)

    #first we get the rotated image.    
    (pivX, pivY) = get_centerpoint(image, angle)
    rotated_image = rotate_image(image, angle)
    ##(h, w) = imageOne.shape[:2]
    (h, w) = rotated_image.shape[:2]
    centerX = w/2
    centerY = h/2
    test_copy = rotated_image.copy()
    
    ##image = cv2.cvtColor(imageOne, cv2.COLOR_BGR2GRAY)

    #we convert image to blob format for face detection
    #blobFromImage(const Mat &image, double scalefactor=1.0,
    # const Size &size=Size(), const Scalar &mean=Scalar(), bool swapRB=true)
    blob = cv2.dnn.blobFromImage(cv2.resize(rotated_image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    #mean: scalar with mean values which are subtracted from channels.
    # Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR 
    # ordering and swapRB is true.

    #pass the blob through the network and obtain the detections and predictions
    print("Computing detections...")
    nnet.setInput(blob)
    detections = nnet.forward();
    
    
    #loop over the detections' channels/depth? (!)(i) what kind of Mat is returned by Net.forward()?.
    # ith detection.
    #what this tells me is that this is at least a 3D array & each face|detections is a 2d array,
    #_ & shape[2] gives the depth, thus the number of 2D arrays, that's why
    #_ the 3rd dimension fetches the ith detection.
    firstA = 0 #True;
    firstB = True;
    # eye_detections = np.zeros(1)
    # roi_color = np.zeros(1)
    
    #print(detections.shape[2])
    print("Image type:")
    print(type(rotated_image))
    
    #Go through the faces detected.
    for i in range(0, detections.shape[2]):
    
        # extract the confidence/probability associated w/ the predictions        
        #mat[0, 0, ith detection, confidence is at index (0, 0, i) 2]
        confidence = detections[0, 0, i, 2]
        
        #filter out weak detections by considering the confidence is greater than
        #_ the min confidence or the greatest detection so far.
        if confidence > (0.5+0.1): #math.max((0.5+0.1), best_confidence):
        
            #get ith detection's bounding box dimensions, w, h, x, y
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #proximity y factor
            proximity_factor = 1.0
            
            if (startY + (endY-startY)/2) >= centerY:
                proximity_factor = 1.2
                
            # cv2.circle(rotated_image, (1, (startY + (endY-startY)/2)), 4,
                    # (255, 255, 255), -1)
            
            # cv2.circle(rotated_image, (1, centerY), 2,
                    # (0, 255, 0), -1)
                    
            # cv2.circle(rotated_image, (1, startY), 4,
                    # (255, 255, 255), -1)
        
            confidence_plus = confidence*100 * ((1 - (abs(centerX-(startX+(endX-startX)))/float(centerX)))*1.5) * (1 - (abs(centerY-(startY+(endY-startY)))/float(centerY)))*proximity_factor* ((endX-startX)/ float(centerX/2)) * ((endY-startY)/ float(centerY/2))
            
            #commented out
            # print("confidence_plus:centerX:{}; thisX:{}".format(centerX, (startX+(endX-startX))))   
            # print("\t1: centerX - thisX:{}:".format(abs(centerX - (startX+(endX-startX)))))
            # print("\t2: (centerX-thisX)/centerX:{}".format(abs(centerX-(startX+(endX-startX)))/float(centerX)))
            # print("\t3: 1- (centerX-thisX)/centerX:{}".format(1 - (abs(centerX-(startX+(endX-startX)))/float(centerX))))
            
            #Convert the confidence score to a percentage out of 100
            text = "{:.2f}%".format(confidence*100)
            text_plus = "{:.2f}%".format(confidence_plus)
            #commented out
            # print("$$$$ Confidence Plus {}:".format(confidence_plus))
            
            #(i) different if. like this if true else this.
            # we reduce the size of the top of the bounding box to make room for the float.
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            #check if this detection is 'better' than our curr best.
            # if best_confidence < confidence:
            if best_confidence < confidence_plus:
                #save the prev best face (2nd best face)
                sec_best_face = best_face
                sec_best_angle = best_angle
                sec_best_confidence = best_confidence
                #Save the new best face & its info, but a clean pic.
                #(!)(?) First rotate image back? and then crop?
                bestX, bestY, bestEndX, bestEndY = startX, startY, endX, endY
                #startX-int(abs((endX-startX)*0.2)), startY-int(abs((endY-startY)*0.2)), endX+int(abs((endX-startX)*0.2)), endY+int(abs((endY-startY)*0.2))
                best_face = rotate_image(image, angle)[bestY:bestEndY, bestX:bestEndX] #startY:endY, startX:endX] #rotated_image[startY:endY, startX:endX] 
                best_confidence = confidence_plus
                best_angle = angle       
                
                foundNew = True
            
            #draw the rectangle around the face
            cv2.rectangle(rotated_image, (startX, startY), (endX, endY),
                (255, 0, 0), 2)
                
            cv2.rectangle(test_copy, (startX, startY), (endX, endY),
                (255, 0, 0), 2)
            
            #draw the confidence on blue background
            cv2.rectangle(rotated_image, (startX, y+10), (startX + ((endX-startX)/4)*3, y-15),
                (255, 0, 0), -1)# Not in cv2: cv2.cv.CV_FILLED)
            #draw on copy.
            cv2.rectangle(test_copy, (startX, y+10), (startX + ((endX-startX)/4)*3, y-15),
                (255, 0, 0), -1)# Not in cv2: cv2.cv.CV_FILLED)
                
            #we get the last face drawn
            #if angle == 120 and firstA:
            firstA+=1
            if firstA == 2: #for testing eye detection & for testing point rotation
                #we get the midpoint of the bounding box & draw a dot there
                # centerX = ((endX-startX)/2)+startX
                # centerY = ((endY-startY)/2)+startY
                # cv2.circle(rotated_image, (centerX, centerY), 14, 
                    # (0, 255, 0), -1)
                ##
                #We detect eyes in the bounding box
                #print("Detecting eyes...")

                #We limit our search to the bounding box where the supposed face was detected.                
                #draw point to rotate.
                #Draws a random circle on top left corner of detected. I guess ith detection.
                ##cv2.circle(rotated_image, (startX, startY), 5,
                    # (0, 255, 0), -1)
                    
                # gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
                # roi_gray = gray_image[startY:endY, startX:endX]
                # roi_color = rotated_image[startY:endY, startX:endX] #we make a variable of certain area of the original image, but we are altering the original image.
                # cv2.imshow("Face", roi_gray)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # eye_detections = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.3)
                # print(type(roi_color))
                # print(roi_color)
                firstA = False
            
            #Draw the confidence of this detection.
            cv2.putText(rotated_image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
                
            cv2.putText(test_copy, text_plus, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
    ##_#######################################################################
    ##End goind through all detections in this angle of image (incl. drawing).
    
    ##Now we show the all the faces detected in this angle of the image.
    
    ##We can indicate the best face we have found so far.
    #If we wanted the best face in this angle,
        #we would reset the best image after every detection.
    
    ##We should draw it based on the curr angle, so points should be rotated.
    #If we only want to draw it when we find it, then we just check if
    #_it's new by comparing w/ prev. Else use boolean.
    if foundNew:
        cv2.rectangle(rotated_image, (bestX, bestY), (bestEndX, bestEndY),
            (0, 200, 255), 3)
        foundNew = False
    
    
    #Show the faces found. Subst. below, commented code for this next line.
    ##
    # show_image_destroy(rotated_image, 'Faces Found in This Angle', True)
    # show_image_destroy(test_copy, 'New Confidences')
    
    # cv2.namedWindow(, cv2.WINDOW_NORMAL)

    # cv2.imshow('Faces Found in This Angle', rotated_image)
    # imWidth = int(rotated_image.shape[0]*0.9)#0.75)
    # imHeight = int(rotated_image.shape[1]*0.9)#0.8)

    # cv2.resizeWindow('Faces Found in This Angle', imHeight, imWidth)
    # #cv2.resizeWindow('Faces Found', testImg1.shape[0]/2.0, testImg1.shape[1]/2.0)
    # # height=testImg1/2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
##_###########################################################################
##End going through angles, so went through all angles of this particular image
    ## in the sequence of images.
#We pretend that we went through the entire sequence, so we have the best image overall.
##best_face is the face upright.
# show_image(best_face, '1st Best_Face&Angle')

##We want an expanded version of the face in order to more accurately find the actual best angle
#make a mat to hold the expanded face roi.
roi_face = rotate_image(image.copy(), best_angle)
##make a copy of prev mat to crop from.
temp_image = roi_face.copy()
# show_image(roi_face, 'Entire Img')

#crop the expanded face.
cropX, cropEndX, cropY, cropEndY = int(bestX-(bestX*0.25)), int(bestEndX+(bestX*0.25)), int(bestY-(bestY*0.25)), int(bestEndY+(bestY*0.25))
roi_face = temp_image[cropY: cropEndY, cropX:cropEndX]
best_face = roi_face

#show the cropped face and the entire image w/ a rect of roi to see if they're matching.
# show_image(roi_face, 'Best Face&Angle Expanded')

##We can either send the face w/ the 1st best_angle applied, or we reset it (angle == 0) , which doesn't make sense.
# roi_face = rotate_image(roi_face, -1*best_angle)
# #face expanded reset
# show_image(roi_face, 'Prev Face Expanded w/o Angle')

# # nx, ny = point_rotation_calc(roi_face, -1*best_angle, (cropX, cropY))#rotate_point(
# # ##(C) (!) Here now
# # nex, ney = point_rotation_calc(roi_face, -1*best_angle, (cropEndX, cropEndY))

# # imt = image.copy()[ny: ney, nx:nex]
# # show_image(imt, 'Matching')



##show rect of roi
# cv2.rectangle(temp_image, (cropX, cropY), (cropEndX, cropEndY), (0, 200, 200), 2)
# show_image(temp_image, 'Face Rect')
# temp_image = rotate_image(temp_image, best_angle)
# show_image(temp_image, 'Expanded image w/ angle')
# cv2.waitKey(0)
# cv2.destroyAllWindows()



(bx, by, bex, bey) = (bestX, bestY, bestEndX, bestEndY)

drawing_copy = roi_face.copy()

best_confidence_original = 0.0
best_angle_original = best_angle


print("Best Confidence Original: {:.2f}%".format(best_confidence))

##Find the best angle for this face.
for angle in it.chain(range(0, -40, -10), range(10, 40, 10)):
    
    rotated_image = rotate_image(roi_face, angle)
    
    # drawing_copy = rotated_image.copy()
    
    # show_image_destroy(rotated_image, "Rotated Image")
    
    (h, w) = rotated_image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(rotated_image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    #mean: scalar with mean values which are subtracted from channels.
    # Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR 
    # ordering and swapRB is true.

    #pass the blob through the network and obtain the detections and predictions
    print("Computing detections...")
    nnet.setInput(blob)
    detections = nnet.forward();
    
    #cycle through the many face detections.    
    # print("Number of detections in this angle: {}".format(detections.shape[2]))
    
    for i in range(0, detections.shape[2]):
        
        # extract the confidence/probability associated w/ the predictions        
        #mat[0, 0, ith detection, confidence is at index (0, 0, i) 2]
        confidence = detections[0, 0, i, 2]
        
        #filter out weak detections by considering the confidence is greater than
        #_ the min confidence or the greatest detection so far.
        
        if confidence > (0.5+0.1): #math.max((0.5+0.1), best_confidence):      
                
            print("\n{:.2f}%".format(confidence*100))
            #get ith detection's bounding box dimensions, w, h, x, y
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            # print("Size of orig Img: {}, {}".format(image.shape[1], image.shape[0]))
            # print("Size of orig Img: {}, {}".format(rotated_image.shape[1], rotated_image.shape[0]))
            (sx, sy, wx, hy) = box.astype("int")
            
            # print("startX: {}, endX: {}, startY: {}, endY: {}".format(sx, sy, wx, hy))            
            #we make sure to get the original unmodified confidence by checking to see if it's at its original angle, and if the box is w/in bounds.
            if angle == 0 and confidence*100 > best_confidence_original and (0 < sx and wx < rotated_image.shape[1]) and (0 < sy and hy < rotated_image.shape[0]):
                best_confidence_original = confidence*100
                print("Best Confidence Original New: {:.2f}%".format(confidence*100))
            # proximity_factor = 1.0
            
            # if (startY + (endY-startY)/2) >= centerY:
                # proximity_factor = 1.2        
        
            ##unaugmented confidence is used here.
            
            # confidence_plus = confidence*100 * ((1 - (abs(centerX-(startX+(endX-startX)))/float(centerX)))*1.5) * (1 - (abs(centerY-(startY+(endY-startY)))/float(centerY)))*proximity_factor* ((endX-startX)/ float(centerX/2)) * ((endY-startY)/ float(centerY/2))
            
            #commented out
            # print("confidence_plus:centerX:{}; thisX:{}".format(centerX, (startX+(endX-startX))))   
            # print("\t1: centerX - thisX:{}:".format(abs(centerX - (startX+(endX-startX)))))
            # print("\t2: (centerX-thisX)/centerX:{}".format(abs(centerX-(startX+(endX-startX)))/float(centerX)))
            # print("\t3: 1- (centerX-thisX)/centerX:{}".format(1 - (abs(centerX-(startX+(endX-startX)))/float(centerX))))
            
            #Convert the confidence score to a percentage out of 100
            text = "{:.2f}%".format(confidence*100)
            
            
            #commented out
            # print("$$$$ Confidence Plus {}:".format(confidence_plus))
            
            #(i) different if. like this if true else this.
            # we reduce the size of the top of the bounding box to make room for the float.
            y = sy - 10 if sy - 10 > 10 else sy + 10
            
            #check if this detection is 'better' than our curr best.
            # if best_confidence < confidence && bounding box w/in bounds (valid):
            if confidence*100 > best_confidence_original and (0 < sx and wx < rotated_image.shape[1]) and (0 < sy and hy < rotated_image.shape[0]):
                
                print("Found a better confidence: {:.2f}%".format(confidence*100))
                #save the prev best face (2nd best face)
                sec_best_face = best_face
                sec_best_angle = best_angle
                sec_best_confidence = best_confidence ##best_confidence_original
                #Save the new best face & its info, but a clean pic.
                #(!)(?) First rotate image back? and then crop?
                bx, by, bex, bey = sx, sy, wx, hy
                
                #startX-int(abs((endX-startX)*0.2)), startY-int(abs((endY-startY)*0.2)), endX+int(abs((endX-startX)*0.2)), endY+int(abs((endY-startY)*0.2))
                best_face = rotated_image[by:bey, bx:bex] #rotated_image #rotate_image(image, best_angle+angle)[bx:bey, bx:bex] #startY:endY, startX:endX] #rotated_image[startY:endY, startX:endX] 
                best_confidence_original = confidence*100
                best_angle = best_angle_original+angle #angle ##best_angle+angle                
            
            #draw the rectangle around the face
            
            print("##Drew rect")
            
            drawing_copy = rotated_image.copy()
            cv2.rectangle(drawing_copy, (sx, sy), (wx, hy),
                (255, 250, 0), 2)
            
            #draw the confidence on blue background
            cv2.rectangle(drawing_copy, (sx, y+10), (sy + ((wx-sx)/4)*3, y-15),
                (255, 250, 0), -1)# Not in cv2: cv2.cv.CV_FILLED)
            
            #Draw the confidence of this detection.
            cv2.putText(drawing_copy, text, (sx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)
            
            ## show_image_destroy(drawing_copy, 'Faces Found in This Angle')
            drawing_copy = rotated_image.copy()
    
    ## For the detections in this rotation.
    ##_#######################################################################
#

##(C) (!)
##Print best_face w/o destroying.
    ##Print image w/ best rotation to match the angle of best_face.
    ##Use temp_image[int(bestY-(bestY*0.25)):int(bestEndY+(bestY*0.25)) , int(bestX-(bestX*0.25)):int(bestEndX+(bestX*0.25))] && bx, by, ... to match
    ##the eyes in rotated image.

##One solution, is after noticing that best_face becomes a crop of the rotate roi_face, which is already a cropped face.
    ##So we can either save best_face as the rotated roi inside the above part, or we save it out here. I think it can just be done inside.

#we locate the center_point of roi_face (pivot)    
roi_copy = roi_face.copy()
cv2.circle(roi_copy, (roi_copy.shape[1]/2, roi_copy.shape[0]/2), 4, (0, 0, 0), -1)
cv2.circle(roi_copy, (roi_copy.shape[1]/2, roi_copy.shape[0]/2), 2, (200, 250, 25), -1)
show_image(roi_copy, "Roi pivot")


##Now we try to match the same in the image, but it will match.
supp_angle = best_angle - best_angle_original
image_roi = rotate_image(image, best_angle_original)
cv2.circle(image_roi, (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)), 4, (0, 0, 0), -1)
cv2.circle(image_roi, (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)), 2, (200, 250, 25), -1)
show_image(image_roi, "Img roi pivot")

##Now we try to rotate the point correctly in roi_face
# # point_rotation_calc(image, angle, point, pivot. piont.
center_r = point_rotation_calc(roi_face, supp_angle, (roi_face.shape[1]/2, roi_face.shape[0]/2))
new_roi = rotate_image(roi_face, supp_angle)

cv2.circle(new_roi, center_r, 4, (0, 0, 0), -1)
cv2.circle(new_roi, center_r, 2, (200, 250, 25), -1)

show_image(new_roi, "Rotated pivot")

##Now we rotate the point try to find it in the other thing
##perhaps find how (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)) goes to the point. True.
##We can try keeping the points of the original rect rotated as well so to find where they align
##The cheap way is the way below. 

new_pt = point_rotation_calc(image_roi, supp_angle, (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)))
im_test = rotate_image(image_roi, supp_angle)
cv2.circle(im_test, new_pt, 4, (0, 0, 0), -1)
cv2.circle(im_test, new_pt, 2, (200, 250, 25), -1)
show_image(im_test, "New point")

#Now we get the translation to apply it to image_roi.
# t_point = get_diff_translation((roi_face.shape[1]/2, roi_face.shape[0]/2), center_r)

# r_center = translate_2D_point((cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)) , t_point)

# im_test = rotate_image(image, best_angle)
# cv2.circle(im_test, r_center, 4, (0, 0, 0), -1)
# cv2.circle(im_test, r_center, 2, (250, 0, 0), -1)
# cv2.circle(im_test, (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)), 4, (0, 0, 0), -1)
# cv2.circle(im_test, (cropX+((cropEndX - cropX)/2) , cropY+((cropEndY - cropY)/2)), 2, (0, 250, 0), -1)

# show_image(im_test, "Aligned point in Image")



##A
# # show_image(roi_face, "New ROI face?")
# #We show the new best face angle and all. Will just be expanded face rotated.
# show_image(best_face, "New Best Face w/ new angle")
# #Trying to match the rotation of the best_face.

# test_image = rotate_image(image, best_angle)
# ##roi_face starts at a different angle.
# test_image2 = rotate_image(roi_face, best_angle - best_angle_original)

# #we figure the center of rotated roi
# (tw, th) = size_rotated_image(roi_face.copy(), best_angle - best_angle_original)
# cv2.circle(test_image2, (tw/2, th/2), 3, (0, 0, 0), -1)
# cv2.circle(test_image2, (tw/2, th/2), 2, (250, 0, 150), -1)

# show_image(test_image2, "Center of roi_face")

# center_o = (cropX+((cropEndX-cropX)/2), cropY+((cropEndY-cropY)/2))
# # cv2.circle(test_image, center_o, 4, (200, 200, 0), -1)


# #Try to get the same center in the entire image.
# # cv2.circle(test_image, (center_o), 3, (0, 0, 0), -1)
# # cv2.circle(test_image, (center_o), 2, (200, 100, 100), -1)
# ##Then I should try to align the centers.
# # show_image(test_image, 'Same center test')

# #Let me try to trace the outline of the roi
# # show_image(roi_face, "ROI")

# roi_outline = rotate_image(image, best_angle_original)
# ##Here Now, trying to get the center correctly.

# ti = rotate_image(image, best_angle)

# # point_rotation_calc(image, angle, point, pivot. piont.
# new_c = point_rotation_calc(roi_outline, best_angle-best_angle_original, center_o)

# cv2.circle(ti, new_c, 3, (0, 0, 0), -1)
# cv2.circle(ti, new_c, 2, (200, 0, 200), -1)

# # cv2.rectangle(roi_outline, (cropX, cropY), (cropEndX, cropEndY), (200, 200, 200), 2)
# show_image(ti, "Center Test")
##A

##(!) vv check if .copy() is necessary in function, if value is modified.
# show_image(rotate_image(roi_face, best_angle-best_angle_original), 'Trying to Match rotation 2')


#cv2.rectangle(best_face, (0, 0),(best_face.shape[0], best_face.shape[1]),(0, 255, 0), 2)  
# cv2.rectangle(test_image, (int(center_o[0]-(tw/2)+bx), int(center_o[1]-(th/2)+by)), (int(center_o[0]-(tw/2)+bex), int(center_o[1]-(th/2)+bey)), 
    # (200, 200, 0), 2)

# show_image(rotate_image(image.copy(), best_angle), "Trying to match rotation.")
# show_image(rotate_image(roi_face, best_angle), "First Rotated after new angle")


cv2.waitKey(0)
cv2.destroyAllWindows()

##

    
    # if best_confidence < confidence_plus:
        # #save the prev best face (2nd best face)
        # sec_best_face = best_face
        # sec_best_angle = best_angle
        # sec_best_confidence = best_confidence
        # #Save the new best face & its info, but a clean pic.
        # #(!)(?) First rotate image back? and then crop?
        # bestX, bestY, bestEndX, bestEndY = startX, startY, endX, endY
        # #startX-int(abs((endX-startX)*0.2)), startY-int(abs((endY-startY)*0.2)), endX+int(abs((endX-startX)*0.2)), endY+int(abs((endY-startY)*0.2))
        # best_face = rotate_image(image, angle)[bestY:bestEndY, bestX:bestEndX] #startY:endY, startX:endX] #rotated_image[startY:endY, startX:endX] 
        # best_confidence = confidence_plus
        # best_angle = angle       
        
        # foundNew = True

#cv2.imwrite('best_face_unrotated.jpg', best_face)

# cv2.namedWindow('Best_Face_Found', cv2.WINDOW_AUTOSIZE)

# cv2.imshow('Best_Face_Found', best_face)

# imWidth = int(best_face.shape[0]*0.9)#0.75)
# imHeight = int(best_face.shape[1]*0.9)#0.8)

# cv2.resizeWindow('Best_Face_Found', imHeight, imWidth)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#Now we detect the eyes in our good image.

#Convert to grayscale so our eye detection works, but we only pass the upper 'half' of the face 
#11/16ths but less than or equal to 3/4ths is a good size.
roi_gray = cv2.cvtColor(best_face[0:(best_face.shape[0]/16)*11, 0:best_face.shape[1]], cv2.COLOR_BGR2GRAY)

show_image_destroy(roi_gray, 'Gray_Face')

##Show gray_face
# # cv2.imshow('Best_Face_Found', best_face)
# cv2.imshow('Gray_Face', roi_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#cv2.rectangle(best_face, (0, 0),(best_face.shape[0], best_face.shape[1]),(0, 255, 0), 2)  

#(!) add scale Factor control, or complete the entire eye detection part.

total_eyes = 0
interim_count_eyes = 0
equalized = False
sf = 1.1

drawing_copy = best_face.copy()
crop_L = False
crop_R = False

min_dist_L = best_face.shape[1]
min_dist_R = best_face.shape[1]
##As a precaution can even have the default values for the eyes here. but not equal so that dist_o != 0.
left_eye = None #(image_copy.shape[1], image_copy.shape[0])
right_eye = None #(image_copy.shape[1], image_copy.shape[0])
# for eye_number in eye_detections
id = 0

while total_eyes != 2:
    #detect eyes
    show_image_destroy(roi_gray, 'GrayScale')
    
    print("\nScale Factor: {}".format(sf))
    eye_detections = eye_detector.detectMultiScale(roi_gray, scaleFactor=sf)
    
    #we find the number of eyes detected.
    if not len(eye_detections):
        print("Zero eyes detected.")
        interim_count_eyes = 0
    else:
        print
        print(eye_detections.shape)
        print("Number Eyes Detected: {}.".format(eye_detections.shape[0]))
        # print("\tTotal_eyes: {}".format(total_eyes))
        interim_count_eyes = eye_detections.shape[0] #eye_detections.ndim
        for (ex, ey, ew, eh) in eye_detections:
        
            cv2.rectangle(drawing_copy, (ex, ey), (ex+ew, ey+eh), (0, 150, 150), 2)
        show_image_destroy(drawing_copy, "Drawing Eyes")
        drawing_copy = roi_gray.copy()
        
    ##Here now.
    ##->Trying w/ eyes initialized to None.
    
    ##Cannot cycle through eyes on the outside, bc if it's 0.
    ##ToDo: make a copy of the face to draw the eyes found. and delete it when its done.

    ##while goAgain:
    #So I need to loop through eyes if at all inside.
    if interim_count_eyes == 2:

        #we do a double check to see if each eye corresponds to the size.
        ##We will check if the eye is left or right I guess I should assign it to correct one in beginning.
        ##commented out. not needed.
        min_dist = roi_gray.shape[1]
        #we decide if it should be L or R eye
        for (ex, ey, ew, eh) in eye_detections:
        
            # cv2.rectangle(drawing_copy, (ex, ey), (ex+ew, ey+eh), (0, 150, 150), 2)
            # show_image_destroy(drawing_copy, "Drawing Eyes")
            # drawing_copy = roi_gray.copy()
            
            ##(!) We don't need to compare distances here, bc 
            #we check if it's on the left side or right side and take only the eyes where they're supposed to be.
            if crop_L is False and crop_R is False:
                if ex+(ew/2) <= best_face.shape[1]/2:
                    ##we can take any, or make sure it's close enough using the epsilon check which is stronger, and then compare distances.
                    ##if distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8)) < L_min_dist:
                        #if left_eye is None: #needs to be done (None) to not redo the same eye in the same place.
                    if distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8)) < .25*best_face.shape[0]: #Epsilon
                        # if left_eye is None:
                        #if this eye is closer to avg_eye_L than the previous eye.
                        this_dist = distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8))
                        if this_dist < min_dist_L:
                            min_dist_L = this_dist
                            left_eye = (ex+(ew/2), ey+(eh/2))
                else:                
                    if distance((ex+(ew/2), ey+(eh/2)), ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8)) < .25*best_face.shape[0]: #Epsilon
                        # if right_eye is None:
                        this_dist = distance((ex+(ew/2), ey+(eh/2)), ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8))
                        if this_dist < min_dist_R:
                            min_dist_R = this_dist
                            right_eye = (ex+(ew/2), ey+(eh/2))                                
                #We let it pick the best eye. 
                ##We handle the case in which 2 'eyes' are detected on the same side: one correct eye, one false positive. If we don't reduce the problem, this will be an endless cycle.
                #we need to converge the problem to a resolution. In this case we reduce the search area to the appropriate half.
                #We found a right eye.
                # if left_eye is None and right_eye is not None:
                    # ##We make the roi to the left half.
                    # roi_gray = roi_gray[0:roi_gray.shape[0], 0:roi_gray.shape[1]/2]
                    # drawing_copy  = roi_gray.copy()
                    # crop_L = True
                # #We found a left eye.
                # elif right_eye is None and left_eye is not None:
                    # roi_gray = roi_gray[0:roi_gray.shape[0], roi_gray.shape[1]/2:roi_gray.shape[1]]
                    # drawing_copy  = roi_gray.copy()
                    # crop_R = True
                    
            elif crop_L: #we cropped to the left side.
                print("Cropped Left")
                #We can assume that only one correct eye was found                
                if distance((ex+(ew/2), ey+(eh/2)), (roi_gray.shape[1]/2, (best_face.shape[0]*3)/8)) < min_dist_L:
                    min_dist_L = distance((ex+(ew/2), ey+(eh/2)), (roi_gray.shape[1]/2, (best_face.shape[0]*3)/8))
                    left_eye = (ex+(ew/2), ey+(eh/2))
                    # total_eyes+=1
                    
            elif crop_R:
                print("Cropped Right")
                if distance((ex+(ew/2), ey+(eh/2)), (roi_gray.shape[1]/2, (best_face.shape[0]*3)/8)) < min_dist_R:
                    min_dist_R = distance((ex+(ew/2), ey+(eh/2)), (roi_gray.shape[1]/2, (best_face.shape[0]*3)/8))
                    right_eye = ((best_face.shape[1]/2)+ex+(ew/2), ey+(eh/2))       
                    
                    # total_eyes+=1
                    
                    # cv2.rectangle(drawing_copy, (ex, ey), (ex+ew, ey+eh), (0, 150, 150), 2)
                    # show_image_destroy(drawing_copy, "R Eye")
                    # drawing_copy = roi_gray.copy()
                    
                    # drawing_copy = best_face.copy()
                    # cv2.rectangle(drawing_copy, ((best_face.shape[1]/2)+ex, ey), ((best_face.shape[1]/2)+ex+ew, ey+eh), (250, 150, 0), 2)
                    # show_image_destroy(drawing_copy, "New Eye")
                    # drawing_copy = roi_gray.copy()
                
        # cv2.rectangle(image_copy, (ex, ey),(ex+ew, ey+eh),(0, 150, 150), 2)
        ##This bit count the number of eyes left after going through multiple eyes.
        ##It also determines if we need to crop L or R. 
        if left_eye is not None and right_eye is not None:
            total_eyes = 2
        elif left_eye is not None or right_eye is not None:
            total_eyes = 1
        ##The following seem to not overlap.
        if left_eye is not None and total_eyes == 1: #we need to focus on the right side.
            roi_gray = roi_gray[0:roi_gray.shape[0], roi_gray.shape[1]/2: roi_gray.shape[1]]
            crop_R = True
        if right_eye is not None and total_eyes == 1: #we need to focus on the right side.
            roi_gray = roi_gray[0:roi_gray.shape[0], 0: roi_gray.shape[1]/2]
            crop_L = True
            
    if interim_count_eyes == 1:
        ##(!) We don't know if we're getting a smaller roi_gray
        
        
        ##if crop_L is False and crop_R is False:
        
        ##We will check if the eye is left or right I guess I should assign it to correct one in beginning.
        ##(!)(!)(!) for all the for loops here, we can just replace w/ a single variable grab. use for all of the instances of (ex, ey, ew, eh).
        if left_eye is None and right_eye is None:
            ##(!)(!) Optimize for loop, just get tuple from array
            print("Found one eye. Both eyes are empty")
            #we decide if it should be L or R eye
            for (ex, ey, ew, eh) in eye_detections:
                #Left side eye
                if ex+(ew/2) < roi_gray.shape[1]/2:
                    left_eye = (ex+(ew/2), ey+(eh/2))
                    roi_gray = roi_gray[0:roi_gray.shape[0], roi_gray.shape[1]/2:roi_gray.shape[1]]
                    crop_R = True
                    drawing_copy  = roi_gray.copy()
                #Right side eye
                else:
                    right_eye = (ex+(ew/2), ey+(eh/2))   
                    roi_gray = roi_gray[0:roi_gray.shape[0], 0:roi_gray.shape[1]/2]
                    crop_L = True
                    drawing_copy  = roi_gray.copy()
                    
        elif left_eye is None:
            print("Found one eye. Left eye is empty")
            ##we can also check if the roi has been reduced in order to 
            ##Optimize (!)(!)
            for (ex, ey, ew, eh) in eye_detections:
                left_eye = (ex+(ew/2), ey+(eh/2))
        elif right_eye is None:
            print("Found one eye. Right eye is empty")
            ##(!)(!) Optimize for loop, just get tuple from array
            if crop_R is False: ##Not cropped.
                for (ex, ey, ew, eh) in eye_detections:
                    right_eye = (ex+(ew/2), ey+(eh/2))
            else:
                for (ex, ey, ew, eh) in eye_detections:
                    right_eye = ((best_face.shape[1]/2)+ex+(ew/2), ey+(eh/2))
        #eyes should never both be none when entering interim_count_eyes.
        total_eyes += interim_count_eyes
        #This means that this eye will always be taken into account.
        
    if interim_count_eyes == 0:
        #Try different scale factors 
        if not equalized:
            equalized = True
            roi_gray = cv2.equalizeHist(roi_gray)
            drawing_copy = roi_gray.copy()
        elif sf < 1.5:
            sf += 0.1
        else:
            #we give our best guess to where they are
            sf = 1.1
            if left_eye is None and right_eye is None:                
                #we decide if it should be L or R eye
                left_eye = (best_face.shape[1]/4, (best_face.shape[0]*3)/8) ##avg_eye_L
                right_eye = ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8) ##avg_eye_R
                total_eyes += 2
            elif left_eye is None:
                print("Doing left_eye")
                left_eye = (best_face.shape[1]/4, (best_face.shape[0]*3)/8) ##avg_eye_L
                cv2.circle(drawing_copy, left_eye, 3, 
                    (0, 0, 0), -1)        
                cv2.circle(drawing_copy, left_eye, 2, 
                    (255, 255, 0), -1)
                    
                show_image_destroy(drawing_copy, 'This Eye')
                drawing_copy = roi_gray.copy()
                
                total_eyes += 1
            elif right_eye is None:
                print("Doing right_eye")
                right_eye = ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8) ##avg_eye_R
                cv2.circle(drawing_copy, right_eye, 3, 
                    (0, 0, 0), -1)
                cv2.circle(drawing_copy, right_eye, 2, 
                    (255, 255, 0), -1)

                show_image_destroy(drawing_copy, 'This Eye')
                drawing_copy = roi_gray.copy()
                
                total_eyes += 1
                
    if interim_count_eyes > 2:
        min_dist = best_face.shape[1]
        
        # if sf < 1.5:
            # sf += 0.1
        
        for (ex, ey, ew, eh) in eye_detections:
            ##Epsilon is a percentage to be combined w/ the width of the 'face' 
            
            print("Eye #: {}".format(id))
            
            print("Dist L : {}".format(distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8))))
            print("Dist R: {}".format(distance((ex+(ew/2), ey+(eh/2)), ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8))))
            print("To beat L: {}".format(min_dist_L)) #.25*best_face.shape[1]))
            print("To beat R: {}".format(min_dist_R)) 
            
            #if w/in epsilon distance of avg_eye_L
            
            cv2.circle(drawing_copy, (ex+(ew/2), ey+(eh/2)), 3, 
                (0, 0, 0), -1)
            cv2.circle(drawing_copy, (ex+(ew/2), ey+(eh/2)), 2, 
                (255, 255, 0), -1)

            show_image_destroy(drawing_copy, "Eye {}".format(id))
            id += 1
            drawing_copy = roi_gray.copy()
            
            if distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8)) < .25*best_face.shape[1]: #Epsilon
                this_dist = distance((ex+(ew/2), ey+(eh/2)), (best_face.shape[1]/4, (best_face.shape[0]*3)/8))
                if this_dist < min_dist_L: #if it's closer than our previous left_eye, we replace it.
                    left_eye = (ex+(ew/2), ey+(eh/2))
                    min_dist_L = this_dist
            #if w/in epsilon distance of avg_eye_R
            elif distance((ex+(ew/2), ey+(eh/2)), ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8)) < .25*best_face.shape[1]: #Epsilon
                this_dist = distance((ex+(ew/2), ey+(eh/2)), ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8))
                if this_dist < min_dist_R:  
                    right_eye = (ex+(ew/2), ey+(eh/2))     
                    min_dist_R = this_dist
            elif sf < 1.5:
                sf += 0.1
            else:
                sf = 1.1
                left_eye = (best_face.shape[1]/4, (best_face.shape[0]*3)/8) ##avg_eye_L
                right_eye = ((best_face.shape[1]*3)/4, (best_face.shape[0]*3)/8) ##avg_eye_R
         
        ##This bit count the number of eyes left after going through multiple eyes.
        ##It also determines if we need to crop L or R. 
        if left_eye is not None and right_eye is not None:
            total_eyes = 2
        elif left_eye is not None or right_eye is not None:
            total_eyes = 1
        ##The following seem to not overlap.
        if left_eye is not None and total_eyes == 1: #we need to focus on the right side.
            roi_gray = roi_gray[0:roi_gray.shape[0], roi_gray.shape[1]/2: roi_gray.shape[1]]
            crop_R = True
        if right_eye is not None and total_eyes == 1: #we need to focus on the right side.
            roi_gray = roi_gray[0:roi_gray.shape[0], 0: roi_gray.shape[1]/2]
            crop_L = True
            
    print("\tTotal_eyes After: {}".format(total_eyes))
    if total_eyes == 2:
        drawing_copy = best_face.copy()
        print("**Width: {}\tHeight:{}".format(best_face.shape[1], best_face.shape[0]))
        print("Right eye: {}".format(right_eye))
        
        cv2.circle(drawing_copy, left_eye, 3, 
            (0, 0, 0), -1)
        cv2.circle(drawing_copy, right_eye, 3, 
             (0, 0, 0), -1)
        cv2.circle(drawing_copy, left_eye, 2, 
            (0, 255, 0), -1)
        cv2.circle(drawing_copy, right_eye, 2, 
             (0, 255, 0), -1)
        show_image_destroy(drawing_copy, "Eyes Found")
##End looking for eyes.        
        
print("\nLeft Eye: {}, {} \t Right Eye: {}, {}".format(left_eye[0], left_eye[1], right_eye[0], right_eye[1]))


im_copy = best_face.copy()

##Draw the left and right eyes' location.
cv2.circle(im_copy, left_eye, 3,
    (250, 250, 250), -1)
cv2.circle(im_copy, left_eye, 2,
    (250, 250, 0), -1)
    
cv2.circle(im_copy, right_eye, 2,
    (250, 250, 0), -1)

##    
show_image(im_copy, "Eyes on Best_Face")

n_leye = ((new_pt[0] - center_r[0]) + bx + left_eye[0], (new_pt[1] - center_r[1]) + by +left_eye[1])
n_reye = ((new_pt[0] - center_r[0]) + bx + right_eye[0], (new_pt[1] - center_r[1]) + by +right_eye[1])

cv2.circle(im_test, ((new_pt[0] - center_r[0]) + bx + left_eye[0], (new_pt[1] - center_r[1]) + by +left_eye[1]), 3,
    (0, 0, 0), -1)
cv2.circle(im_test, ((new_pt[0] - center_r[0]) + bx + left_eye[0], (new_pt[1] - center_r[1]) + by +left_eye[1]), 2,
    (250, 250, 250), -1)    
cv2.circle(im_test, ((new_pt[0] - center_r[0]) + bx + right_eye[0], (new_pt[1] - center_r[1]) + by +right_eye[1]), 3,
    (0, 0, 0), -1) 
cv2.circle(im_test, ((new_pt[0] - center_r[0]) + bx + right_eye[0], (new_pt[1] - center_r[1]) + by +right_eye[1]), 2,
    (250, 250, 250), -1)      


show_image(im_test, "Eyes on Image")


# if not eye_detections: #any(map(len, eye_detections)): #passes if any of the contained items are not empty
    # interim_count_eyes = 0
# else:
    # print("Eye Shape")
    # print(eye_detections.ndim)


# image_copy = best_face.copy()

# # eye_test_copy = rotate_image(image, best_angle)

# ##show entire image, rotated & with eyes.
# rimage = rotate_image(image, best_angle)  

# cv2.rectangle(rimage, (cropX, cropY), (cropEndX, cropEndY),
    # (200, 0, 200), 2)

# cv2.circle(rimage, (left_eye[0]+cropX, left_eye[1]+cropY), 3,
    # (250, 250, 250), -1)
# cv2.circle(rimage, (left_eye[0]+cropX, left_eye[1]+cropY), 2,
    # (0, 250, 250), -1)
    
# cv2.circle(rimage, (right_eye[0]+cropX, right_eye[1]+cropY), 3,
    # (250, 250, 250), -1)
# cv2.circle(rimage, (right_eye[0]+cropX, right_eye[1]+cropY), 2,
    # (0, 250, 250), -1)    
    
# show_image(rimage, "Im_Rotation")
cv2.waitKey(0)
cv2.destroyAllWindows()
##

######(!) As a precaution can even have the default values for the eyes here. but not equal so that dist_o != 0.
# # left_eye = (image_copy.shape[1], image_copy.shape[0])
# # right_eye = (image_copy.shape[1], image_copy.shape[0])

##first i'm gonna print the shape to see which one is the number detections.
##I will assign this number ot a var. We then decide what to do based on the number of eyes detected.
##We can also use scale factor. (!) Look for instances where the shape is used wrong. Print to check.

##commented out.
# # for (ex, ey, ew, eh) in eye_detections:
    # # ##commented out
    # # # print('Shape')
    # # # print(eye_detections.shape)
    # # #if the eye detected is more left than the left one we have.
    # # if ex < leftEye[0]:
        # # rightEye = leftEye
        # # leftEye = (ex+(ew/2), ey+(eh/2))
    # # else:
        # # rightEye = (ex+(ew/2), ey+(eh/2))
    
    # # cv2.rectangle(image_copy, (ex, ey),(ex+ew, ey+eh),(0, 150, 150), 2)        

##We show the eyes we detected
# # cv2.circle(image_copy, leftEye, 2, 
     # # (0, 255, 0), -1)
# # cv2.circle(image_copy, rightEye, 2, 
     # # (0, 255, 0), -1)
     
# cv2.circle(eye_test_copy, (leftEye[0]+bestX, leftEye[1]+bestY), 2, 
     # (0, 255, 0), -1)
# cv2.circle(eye_test_copy, (rightEye[0]+bestX, rightEye[1]+bestY), 2, 
     # (0, 255, 0), -1)
     
# show_image_destroy(eye_test_copy, 'Testing')

##commented out . To show where it detected?     
#cv2.rectangle(image_copy, (0, 0),(image_copy.shape[1], image_copy.shape[0]),(0, 150, 150), 5)

##show_image_destroy(image_copy, 'Eyes_Detected')

# cv2.namedWindow('Eyes_Detected', cv2.WINDOW_NORMAL)

# cv2.imshow('Eyes_Detected', image_copy)
# imWidth = int(image_copy.shape[0]*0.9)#0.75)
# imHeight = int(image_copy.shape[1]*0.9)#0.8)

# cv2.resizeWindow('Eyes_Detected', imHeight, imWidth)
#cv2.resizeWindow('Eyes_Detected', image_copy.shape[0]/2.0, image_copy.shape[1]/2.0)
# height=image_copy/2)

##For now, keep the unaltered image.
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # print("Width:")
# # print(image_copy.shape[0])
# # print("Height:") 
# # print(image_copy.shape[1])

# # get the direction
# #eye_direction = (rightEye[0] - leftEye[0], rightEye[1] - leftEye[1])
# # calc rotation angle in radians
# #rotation = math.atan2(float(eye_direction[1]),float(eye_direction[0]))
# ## We check which is left and which is right (?) (!) no, w is always more right than x.
# ####
# #rotation_degrees = math.degrees(rotation)

#cropped_face = transform_image(best_face, rotation, leftEye, rightEye)
##cropped_face = transform_image(best_face, leftEye, rightEye, (.3, .3)) #( , 0.35) good for height #test (.3, .25)

##cropped_face = transform_image(eye_test_copy, (left_eye[0]+bestX, left_eye[1]+bestY), (right_eye[0]+bestX, right_eye[1]+bestY), (.3, .3))
##cropped_face = transform_image(eye_test_copy, (left_eye[0]+bx+bestX, left_eye[1]+by+bestY), (right_eye[0]+bx+bestX, right_eye[1]+by+bestY), (.3, .3))

# cropped_face = transform_image(fimage, left_eye, right_eye, (.3, .3))
cropped_face = transform_image(im_test, n_leye, n_reye, (.3, .3))

##(!)(A)(i) The prob is that w/ any of the percentages, we can't get the top of the head and the bottom of the head. Perhaps I can use dist_o to find a ratio
##for how much above and below the bounding box should be to encompass the top of the head as well. 
#cropped_face = rotate_image(best_face, rotation_degrees, leftEye)
#cropped_face = CropFace(best_face, eye_left=leftEye, eye_right=rightEye, offset_pct=(0.2,0.2), dest_sz=(300,350))#.save("arnie_10_10_200_200.jpg")
####

show_image_destroy(cropped_face, 'Final Image')


##//Now we crop and normalize the face.

##Show the image?     
# # cv2.namedWindow('Best_Face_Found', cv2.WINDOW_NORMAL)
# # #11/16ths but less than or equal to 3/4ths. 

# # cv2.imshow('Best_Face_Found', best_face[0:(best_face.shape[1]/16)*11, 0:best_face.shape[0]])

# # # imWidth = int(best_face.shape[0]*0.9)#0.75)
# # # imHeight = int(best_face.shape[1]*0.9)#0.8)

# # # cv2.resizeWindow('Best_Face_Found', imHeight, imWidth)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

    
    # for (ex, ey, ew, eh) in eye_detections:
        # cv2.rectangle(roi_color, (ex, ey),(ex+ew, ey+eh),(0, 255, 0), 2)        
    # #We show the eyes we detected
    # cv2.namedWindow('Faces Found', cv2.WINDOW_NORMAL)

    # cv2.imshow('Faces Found', rotated_image)
    # imWidth = int(rotated_image.shape[0]*0.9)#0.75)
    # imHeight = int(rotated_image.shape[1]*0.9)#0.8)

    # cv2.resizeWindow('Faces Found', imHeight, imWidth)
    # #cv2.resizeWindow('Faces Found', testImg1.shape[0]/2.0, testImg1.shape[1]/2.0)
    # # height=testImg1/2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
    
    ##Now we rotate the point back
    #Now we rotate the point and see if it matches. (!) prob need to use the same rotation of the rotationMatrix[0,2],[1,2] used to rotate the image, in reverse angle
    # # if angle == 120:
        # # #get rotated point (reverse)
        # # centerX, centerY = rotate_point_2((imWidth/2, imHeight/2), (centerX, centerY), angle) #rotate_point((centerX, centerY), angle*-1, (pivX, pivY))
        # # #draw the point on the original image
        # # cv2.circle(image, (centerX, centerY), 25, 
            # # (0, 0, 255), -1)
        
        # # cv2.namedWindow('Faces Found', cv2.WINDOW_NORMAL)
        
        # # cv2.imshow('Faces Found', image)
        # # imWidth = int(image.shape[0]*0.9)#0.75)
        # # imHeight = int(image.shape[1]*0.9)#0.8)

        # # cv2.resizeWindow('Faces Found', imHeight, imWidth)
        # # #cv2.resizeWindow('Faces Found', testImg1.shape[0]/2.0, testImg1.shape[1]/2.0)
        # # # height=testImg1/2)

        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        # # firstB = False

    #################################
   
# #Let's show the best face, let's see if we even need to rotate
# cv2.namedWindow('Faces Found', cv2.WINDOW_NORMAL)

# cv2.imshow('Faces Found', best_face)

# imWidth = int(best_face.shape[0]*0.9)#0.75)
# imHeight = int(best_face.shape[1]*0.9)#0.8)

# cv2.resizeWindow('Faces Found', imHeight, imWidth)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


