import math
import os
import cv2
import numpy as np
from pythonRLSA import rlsa
def spLines(num):
    imNum = str(num)
    image = cv2.imread( 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum + '/sp/specialPara.jpg', cv2.IMREAD_GRAYSCALE)
    (thresh, binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_bw = binary.copy()
    (contours, _) = cv2.findContours(~im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    binC1 = binary.copy()
    w = image.shape[1]
    w2 = math.floor(w / 2)
    rlw = rlsa.rlsa(binC1, True, False, w2)  # horizontal
    rl_copy = rlw.copy()

    # histogram
    rl_copy[rl_copy == 0] = 1  # Convert black spots to ones
    rl_copy[rl_copy == 255] = 0  # Convert white spots to zeros
    hist = np.sum(rl_copy, axis=1)
    # draw hist
    m = max(hist)
    height, width = image.shape
    blankImage = np.ones(image.shape[:2], dtype="uint8") * 255
    for row in range(height):
        cv2.line(blankImage, (0, row), (int(hist[row] * width / m), row), 0, 1)

    # values of mid point
    startArray = []
    p = math.floor(width / 10)
    for i in range(height - 1):
        startArray.append(blankImage[i, p])

    def mid_point(valArray):
        indexes = []
        first_one = -1
        for i in range(len(valArray)):
            if first_one == -1 and valArray[i] == 255:
                first_one = i
            elif first_one > -1 and valArray[i] == 0:
                indexes.append((first_one + i - 1) // 2)
                first_one = -1
        if first_one > -1:
            indexes.append((first_one + len(valArray) - 1) // 2)
        return indexes

    mid_indexes = mid_point(startArray)

    # clean indexes
    gap = 0
    count = 0
    for g in range(0, len(mid_indexes) - 1):
        gap = gap + mid_indexes[g + 1] - mid_indexes[g]
        count = count + 1
    mean_gap = math.ceil(gap / count)

    val = math.floor(mean_gap / 4)
    cleanArray = []
    for j in range(0, len(mid_indexes) - 1):
        if (mid_indexes[j + 1] - mid_indexes[j] > val):
            cleanArray.append(mid_indexes[j])

    if (mid_indexes[-1] - mid_indexes[-2] > val):
        cleanArray.append(mid_indexes[-1])


    # save lines into images
    colList = []
    v = 0
    while (v < len(cleanArray) - 1):
        colImg = []
        colImg = np.array(colImg, dtype="object")
        colImg = image[cleanArray[v]:cleanArray[v + 1], 0:w]
        colList.insert(v, colImg)
        v = v + 1
    l = len(colList)
    #
    path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/spParaLines'
    os.mkdir(path)
    for i in range(l):
        cv2.imwrite(os.path.join(path, str(i) + '.jpg'), colList[i])


