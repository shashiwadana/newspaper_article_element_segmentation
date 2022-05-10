import cv2
import numpy as np
from pythonRLSA import rlsa
from matplotlib import pyplot as plt
import math
import os
from glob import glob

def words(imNum, folderName):
    num = str(imNum)
    folder = folderName
    path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/' + folder + 'words/'
    os.mkdir(path)
    roilist = []
    imgs = [cv2.imread(fn, cv2.IMREAD_GRAYSCALE) for fn in glob('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num +'/'+ folder + '/*.jpg')]
    for q in range(len(imgs)):
        image = cv2.imread('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num +'/'+ folder + '/'+ str(q) +'.jpg', cv2.IMREAD_GRAYSCALE)
        (ret, binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_bw = binary.copy()
        (contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        widths = [cv2.boundingRect(contour)[2] for contour in contours]
        avgwidth = sum(widths) / len(widths)  # average height
        avgWidthInt = math.ceil(avgwidth)
        heights = [cv2.boundingRect(contour)[3] for contour in contours]  # collecting heights of each contour
        avgheight = math.ceil(sum(heights) / len(heights))  # average height

        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            cv2.drawContours(binary, [c], -1, 0, -1)

        binC1 = binary.copy()
        rlw = rlsa.rlsa(binC1, True, False, avgWidthInt)
        rlh = rlsa.rlsa(rlw, False, True, avgheight)
        rl_copy = rlh.copy()

        # vertical projection
        rl_copy[rl_copy == 0] = 1  # Convert black spots to ones
        rl_copy[rl_copy == 255] = 0  # Convert white spots to zeros
        hist = np.sum(rl_copy, axis=0)
        width = image.shape[1]
        mask = np.ones(image.shape[:2], dtype="uint8") * 255

        height = image.shape[0]
        width = image.shape[1]

        # save words
        startIndex = 0
        endIndex = 0
        inBlock = False
        j = 0

        for i in range(width):
            if ((inBlock == False) and hist[i] != 0):
                inBlock = True
                startIndex = i


            elif (hist[i] == 0 and inBlock):
                endIndex = i
                inBlock = False
                dens = 0
                count = 0
                for k in range(startIndex, endIndex):
                    dens = dens + hist[k]
                    count = count + 1
                area = count * width
                pix = math.floor(area * 0.01)
                if (dens > pix):
                    roiImage = []
                    roiImage = image[0: height, startIndex: endIndex + 1]
                    roilist.append(roiImage)
                    #roilist.append(whiteImage)
                    j = j + 2

                else:
                    continue
    l = len(roilist)
    for i in range(l):
        cv2.imwrite(os.path.join(path, str(i) + '.jpg'), roilist[i])


    return True


