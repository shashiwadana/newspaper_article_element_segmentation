from glob import glob
import cv2 as cv
import numpy as np
import os

def charSegment(imNum,folderName, fName):
    print("characters segmenting")
    num = str(imNum)
    folder = folderName
    path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/CharacterElements/' + num + '/' + fName + '/'
    os.mkdir(path)
    imgs = [cv.imread(fn, cv.IMREAD_GRAYSCALE) for fn in glob('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/' + folder + '/' + '/*.jpg')]
    roilist = []
    whiteImage = np.ones((20, 20), dtype="uint8") * 255
    for q in range(len(imgs)):
        roilist.append(whiteImage)
        img = cv.imread('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/' + folder + '/' + str(q) + '.jpg',
                        cv.IMREAD_GRAYSCALE)
        ret3, th3 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # vertical pro
        vp = th3.copy()
        vp[vp == 0] = 1
        vp[vp == 255] = 0
        vpHst = np.sum(vp, axis=1)

        th = 1
        H, W = img.shape[:2]
        uppers = [y for y in range(H - 1) if vpHst[y] <= th and vpHst[y + 1] > th]
        lowers = [y for y in range(H - 1) if vpHst[y] > th and vpHst[y + 1] <= th]

        if (len(uppers) == 0):
            start = 0
        else:
            start = min(uppers)
        if (len(lowers) == 0):
            end = H
        else:
            end = max(lowers)

        if (start > end):
            start = 0
            end = H

        # horizontal
        proj = th3.copy()
        proj[proj == 0] = 1
        proj[proj == 255] = 0
        hist = np.sum(proj, axis=0)
        height, width = th3.shape

        # save chars
        startIndex = 0
        endIndex = 0
        inBlock = False
        j = 0
        # pix = math.floor(height * 0.01)
        for i in range(width):
            if ((inBlock == False) and hist[i] != 0):
                inBlock = True
                startIndex = i

            elif (hist[i] == 0 and inBlock):
                endIndex = i
                inBlock = False
                roiImage = img[start:end, startIndex: endIndex + 1]
                # resized = cv.resize(roiImage, (50,50), interpolation=cv.INTER_AREA)
                roilist.append(roiImage)
                j = j + 2

    l = len(roilist)
    if(l>0):
        for k in range(l):
            cv.imwrite(os.path.join(path, str(k) + '.jpg'), roilist[k])


