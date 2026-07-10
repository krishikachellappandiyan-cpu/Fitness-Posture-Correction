# ==================== IMPORT PACKAGES =================

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt 

from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential

import cv2
from skimage.io import imshow

import os
import argparse
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# ============= READ INPUT VIDEO ================

# Open the video file.
filename = askopenfilename()
cap = cv2.VideoCapture(filename)
Frames_all = []
# Loop over the frames in the video.
while True:
    # Read the next frame from the video.
    ret, frame = cap.read()

    # If the frame is not read successfully, break from the loop.
    if not ret:
        break

    # Convert the frame to grayscale.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame.
    cv2.imshow('Frame', frame)
    Frames_all.append(frame)
    # Wait for a key press.
    key = cv2.waitKey(1)

    # If the key pressed is `q`, break from the loop.
    if key == ord('q'):
        break

# Close the video file.
cap.release()

# Destroy all windows created by OpenCV.
cv2.destroyAllWindows()


# ===================== CONVERT VIDO INTO FRAMES ===================


Testfeature = []

for iiij in range(0,len(Frames_all)):
    
    img1 = Frames_all[iiij]
    
    plt.imshow(img1)
    plt.title('ORIGINAL IMAGE')
    plt.show()
    
    #
    # PRE-PROCESSING
    
    h1=512
    w1=512
    
    dimension = (w1, h1) 
    resized_image1 = cv2.resize(img1,(h1,w1))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image1)
    plt.show()
    
    
    # ===== FEATURE EXTRACTION ====
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(resized_image1)
    median_val = np.median(resized_image1)
    var_val = np.var(resized_image1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)    
    
    
    # ==== LBP =========
    
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
       
          
    def find_pixel(imgg, center, x, y):
        new_value = 0
        try:
            if imgg[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value
       
    # Function for calculating LBP
    def lbp_calculated_pixel(imgg, x, y):
        center = imgg[x][y]
        val_ar = []
        val_ar.append(find_pixel(imgg, center, x-1, y-1))
        val_ar.append(find_pixel(imgg, center, x-1, y))
        val_ar.append(find_pixel(imgg, center, x-1, y + 1))
        val_ar.append(find_pixel(imgg, center, x, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y))
        val_ar.append(find_pixel(imgg, center, x + 1, y-1))
        val_ar.append(find_pixel(imgg, center, x, y-1))
        power_value = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_value[i]
        return val
       
       
    height, width, _ = img1.shape
       
    img_gray_conv = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
       
    img_lbp = np.zeros((height, width),np.uint8)
       
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
    
    plt.imshow(img_lbp, cmap ="gray")
    plt.title("LBP")
    plt.show()    
        
    # =============== GLCM ===================

    
    from skimage.feature import graycomatrix, graycoprops
    
    # -- FEATURE EXTRACTION
    # Face
    
    PATCH_SIZE = 21
    
    image = resized_image1[:,:,0]
    image = cv2.resize(image,(768,1024))
    
    # select some patches from foreground and background
    
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])
    
    # select some patches from sky areas of the image
    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
    sky_patches = []
    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])
    
    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (grass_patches + sky_patches):
        glcm = graycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)
        xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(graycoprops(glcm, 'correlation')[0, 0])
    
    # create the figure
    fig = plt.figure(figsize=(8, 8))
    
    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in grass_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    for (y, x) in sky_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')
    
    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
            label='Region 1')
    ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
            label='Region 2')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Feature Points')
    ax.legend()
    
    # display the image patches
    for i, patch in enumerate(grass_patches):
        ax = fig.add_subplot(3, len(grass_patches), len(grass_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Region 1 %d' % (i + 1))
    
    for i, patch in enumerate(sky_patches):
        ax = fig.add_subplot(3, len(sky_patches), len(sky_patches)*2 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Region 2 %d' % (i + 1))
    
    
    # display the patches and plot
    fig.suptitle('co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    
    sky_patches0 = np.mean(sky_patches[0])
    sky_patches1 = np.mean(sky_patches[1])
    sky_patches2 = np.mean(sky_patches[2])
    sky_patches3 = np.mean(sky_patches[3])
    
    
    Glcm_fea2 = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
    
    Testfeature.append(Glcm_fea2)
