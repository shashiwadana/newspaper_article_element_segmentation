import os
import cv2
import numpy as np
from pythonRLSA import rlsa
import math
from glob import glob
from matplotlib import pyplot as plt
from itertools import zip_longest

def titleCont(num):
    print("Title and Content segmenting...")
    imNum = str(num)
    imageC = cv2.imread("C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/" + imNum + "/pp.jpg", cv2.IMREAD_GRAYSCALE)
    (thresh, binary) = cv2.threshold(imageC, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mWidth = imageC.shape[1]
    imgs = [cv2.imread(fn, cv2.IMREAD_COLOR) for fn in glob('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum +'/rawCols' + '/*.jpg')]
    all_endIndexes = []
    width_queue = []
    for q in range(len(imgs)):
        image = imgs[q]  # reading the image
        width_queue.append(image.shape[1])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting to grayscale image
        (thresh, im_bw) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # converting to binary image
        mask = np.ones(image.shape[:2],dtype="uint8") * 255  # create blank image of same dimension of the original image
        (contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = [cv2.boundingRect(contour)[3] for contour in contours]  # collecting heights of each contour
        avgheight = math.floor(sum (heights)/ len(heights))  # average height
        # finding the larger text
        for c in contours:
            [x, y, w, h] = cv2.boundingRect(c)
            if h > 1.8 * avgheight and h < 3 * w:
                cv2.drawContours(mask, [c], -1, 0, -1)
                # cv2.rectangle(mask,(x,y),(x+w,y+h),0,-1)

        mask_copy = mask.copy()
        hz = math.ceil(image.shape[1] / 2)
        rl = rlsa.rlsa(mask_copy, True, False, hz)
        dilatd = cv2.dilate(rl, np.ones((1, 3)), 1)

        # part2
        mask2 = np.ones(image.shape[:2], dtype="uint8") * 255
        (contours2, _) = cv2.findContours(~dilatd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights2 = [cv2.boundingRect(contour2)[3] for contour2 in contours2]
        # print("heights2",heights2)
        # print("max height",max(heights2))
        maxHght = math.floor(max(heights2) * 0.6)
        # print(maxHght)
        h1Arr = []
        for c2 in contours2:
            [x1, y1, w1, h1] = cv2.boundingRect(c2)
            if h1 > maxHght:
                h1Arr.append(h1)
                cv2.rectangle(mask2, (x1, y1), (x1 + w1, y1 + h1), 0, -1)

        mCopy = mask2.copy()
        hz = math.ceil(image.shape[1] / 4)
        rl = rlsa.rlsa(mCopy, True, False, hz)

        # projrction
        rl[rl == 0] = 1  # Convert black spots to ones
        rl[rl == 255] = 0  # Convert white spots to zeros
        hist = np.sum(rl, axis=1)

        width = image.shape[1]
        pix = math.floor(width * 0.01)
        start = 0
        end = 0
        inBlock = False
        endIndexes = []

        # get line heights
        heightC = imageC.shape[0]
        hgt = math.floor(heightC * 0.45)
        for i in range(0, hgt):
            if ((inBlock == False) and hist[i] >= pix):
                inBlock = True
                start = i
                cv2.line(mask, (0, start), (width, start), 0, 2)

            elif (hist[i] < pix and inBlock):
                end = i
                inBlock = False
                cv2.line(mask, (0, end), (width, end), 0, 2)
                endIndexes.append(end)

        all_endIndexes.append(endIndexes)  # append last index of every line
    #cv2.imwrite('e.jpg', mask)
    out = [max(t) for t in zip_longest(*all_endIndexes, fillvalue=float("-inf"))]  # get max of each identified row

    if(len(out) > 0):
        mMax = max(out)
        point = []
        for le in range(len(all_endIndexes)):  # iterate through columns
            if (len(all_endIndexes[le]) == 0 or max(all_endIndexes[le]) > mMax - 10):
                point.append(mMax)
            else:
                point.append(min(out, key=lambda x: abs(x - max(all_endIndexes[le]))))

            # clean point array to remove points too far
        pointThresh = math.ceil(heightC * 0.2)
        for t in range(len(point) - 1):
            if (point[t + 1] - point[t] > pointThresh):
                point[t + 1] = point[t]
            elif (point[t] - point[t + 1] > pointThresh):
                point[t] = point[t + 1]

        xVal = []
        s = 0
        for x in range(len(width_queue)):
            xVal.append(s)
            s = s + width_queue[x]
        xVal.append(imageC.shape[1])

        Title = binary.copy()
        # save content as seperate images
        for v in range(0, len(xVal) - 1):
            Title[point[v]:, xVal[v]:xVal[v + 1]] = 255

        # whole content
        titlecopy = Title.copy()
        contentInv = cv2.bitwise_xor(binary, titlecopy)
        contentImage = cv2.bitwise_not(contentInv)

        pointMin = min(point)
        newCont = contentImage[pointMin:, 0:mWidth]

        pathT = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum
        cv2.imwrite(os.path.join(pathT, 'Title.jpg'), Title)
        pathCont = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum
        cv2.imwrite(os.path.join(pathCont, 'Content.jpg'), contentImage)
        numColumn = len(imgs)

        cv2.namedWindow('TitleArea', cv2.WINDOW_NORMAL)
        cv2.imshow("TitleArea", Title)
        cv2.namedWindow('ContentArea', cv2.WINDOW_NORMAL)
        cv2.imshow("ContentArea", contentImage)
        cv2.waitKey(0)

        return numColumn, xVal, point

    else :
        #exit()
        print('please add clear image')