import cv2
import numpy as np
import math
import os
from matplotlib import pyplot as plt

def subTitle(imNum):
    print("Title and sub title lines segmenting...")
    sTitle = False
    num = str(imNum)
    image = cv2.imread('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num +'/Title.jpg', cv2.IMREAD_GRAYSCALE)
    (ret, binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = binary.copy()
    (contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        cv2.drawContours(binary, [c], -1, 0, -1)
    # cv2.imwrite('b.jpg',binary)
    bincopy = binary.copy()
    bincopy[bincopy == 0] = 1  # Convert black spots to ones
    bincopy[bincopy == 255] = 0  # Convert white spots to zeros
    hist = np.sum(bincopy, axis=1)

    width = image.shape[1]
    inBlock = False
    endIndexes = []
    # get line heights
    pix = math.floor(width * 0.01)
    for i in range(0, len(hist)):
        if ((inBlock == False) and hist[i] >= pix):
            inBlock = True
            start = i
            cv2.line(binary, (0, start), (width, start), 0, 2)
            endIndexes.append(start)
        elif (hist[i] < pix and inBlock):
            end = i
            inBlock = False
            cv2.line(binary, (0, end), (width, end), 0, 2)
            endIndexes.append(end)

    gap = 0
    count = 0
    for j in range(0, len(endIndexes) - 1):
        gap = gap + endIndexes[j + 1] - endIndexes[j]
        count = count + 1
    avggap = math.floor(gap / count)
    clean = []
    if(len(endIndexes)==2):
        clean.append(endIndexes[0])
        clean.append(endIndexes[1])
    else:
        for k in range(0, len(endIndexes) - 1):
            if (endIndexes[k + 1] - endIndexes[k] > avggap):
                clean.append(endIndexes[k])
                clean.append(endIndexes[k + 1])


    title = []
    subtitle = []
    gapArray = []
    r = 0
    while (r < len(clean)):
        gapArray.append(clean[r + 1] - clean[r])
        r = r + 2

    gapThresh = math.floor(min(gapArray) * 1.5)
    l = 0
    while (l < len(clean)):
        if (clean[l + 1] - clean[l] > gapThresh):
            title.append(clean[l])
            title.append(clean[l + 1])
        else:
            subtitle.append(clean[l])
            subtitle.append(clean[l + 1])
        l = l + 2

    # check title empty then append subtitle as title
    if (len(title) == 0):
        title = subtitle.copy()
        subtitle.clear()
    # save images
    colList = []
    v = 0
    while (v < len(title)):
        colImg = []
        colImg = np.array(colImg, dtype="object")
        colImg = image[title[v]:title[v + 1], 0:width]
        colList.insert(v, colImg)
        v = v + 2
    l = len(colList)
    #
    path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/titleLines'
    os.mkdir(path)
    for i in range(l):
        cv2.imwrite(os.path.join(path, str(i) + '.jpg'), colList[i])
        plt.subplot(3, 3, i + 1), plt.imshow(colList[i],'gray')
        plt.xticks([]), plt.yticks([])
        plt.title('Title')
        plt.show()


    # subTitle
    if (len(subtitle) != 0):
        sTitle = True
        subList = []
        q = 0
        while (q < len(subtitle)):
            subImg = []
            subImg = np.array(subImg, dtype="object")
            subImg = image[subtitle[q]:subtitle[q + 1], 0:width]
            subList.insert(q, subImg)
            q = q + 2
        z = len(subList)
        #
        path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/subTitleLines'
        os.mkdir(path)
        for b in range(z):
            cv2.imwrite(os.path.join(path, str(b) + '.jpg'), subList[b])
            plt.subplot(2, 2, b + 1), plt.imshow(subList[b],'gray')
            plt.xticks([]), plt.yticks([])
            plt.title('Sub Title')
            plt.show()

    cv2.namedWindow('detectedMargins', cv2.WINDOW_NORMAL)
    cv2.imshow('detectedMargins', binary)
    cv2.waitKey(0)
    return sTitle