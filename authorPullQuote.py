import math
import operator
import os
from glob import glob
import cv2
import numpy as np
from pythonRLSA import rlsa

def authHighlight(num):
    print("author name and highlights segmenting...")
    path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/columnLines'
    os.mkdir(path)
    pathHl = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/Highlights'
    os.mkdir(pathHl)
    imgs = [cv2.imread(fn, cv2.IMREAD_GRAYSCALE) for fn in glob('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/columns/' + '/*.jpg')]
    authLine = 0
    highLine = 0
    resp = [0,0]
    colList = []
    for q in range(len(imgs)):
        image = cv2.imread('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/columns/' +str(q) + '.jpg', cv2.IMREAD_GRAYSCALE)
        (thresh, binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        im_bw = binary.copy()
        hBinary = binary.copy()
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

        cleanArray = []
        # draw hist
        m = max(hist)
        if(m > 0):
            height, width = image.shape
            blankImage = np.ones(image.shape[:2], dtype="uint8") * 255
            for row in range(height):
                cv2.line(blankImage, (0, row), (int(hist[row] * width / m), row), 0, 1)

            # values of mid point
            startArray = []
            p = math.floor(width / 10)
            for i in range(height - 1):
                startArray.append(blankImage[i, p])

            # print(startArray)
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
            if (count > 0):
                mean_gap = math.ceil(gap / count)

                val = math.floor(mean_gap / 4)

                for j in range(0, len(mid_indexes) - 1):
                    if (mid_indexes[j + 1] - mid_indexes[j] > val):
                        cleanArray.append(mid_indexes[j])

                if (mid_indexes[-1] - mid_indexes[-2] > val):
                    cleanArray.append(mid_indexes[-1])

            fullLen = len(imgs) - 1
            noLines = len(cleanArray) - 1
            author = False
            #author/ref segmentation
            if (q == fullLen): #last column
                if (len(cleanArray) > 2):
                    # extract last line
                    authL = image[cleanArray[-2]:cleanArray[-1], 0:w]
                    (threshAuth, bAuth) = cv2.threshold(authL, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    authorLine = bAuth.copy()

                    # calculate white blob

                    authorLine[authorLine == 0] = 1  # Convert black spots to ones
                    authorLine[authorLine == 255] = 0  # Convert white spots to zeros
                    authHist = np.sum(authorLine, axis=0)  # vertical projection

                    inBlock = False
                    blob = 0
                    bloSize = {}
                    for i in range(len(authHist)):
                        if (authHist[i] == 0):
                            blob = blob + 1

                        elif (authHist[i] > 0):
                            bloSize[i] = blob
                            blob = 0

                    #
                    maxBlob = max(bloSize.values())
                    ind = list(bloSize.keys())[list(bloSize.values()).index(maxBlob)]

                    authThresh = math.floor(len(authHist) * 0.20)
                    pos = math.floor(len(authHist) * 0.4)
                    # author = False
                    if (maxBlob >= authThresh and ind < pos):  # maximu white blob is in the first 40% of the width and higher than 20% widths
                        author = True

                    # remove author line from content
                    noLines = 0
                    if (author == True):
                        noLines = len(cleanArray) - 2

                    else:
                        noLines = len(cleanArray)-1
                    #

            # save lines into images

            v = 0
            while (v < noLines):
                colImg = []
                colImg = np.array(colImg, dtype="object")
                colImg = image[cleanArray[v]:cleanArray[v + 1], 0:w]
                colList.append(colImg)
                v = v + 1

            if (author == True):
                resp[0] = 1
                pathAuth = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/authLines'
                os.mkdir(pathAuth)
                cv2.imwrite(os.path.join(pathAuth, '0.jpg'), bAuth)
                cv2.namedWindow('author', cv2.WINDOW_NORMAL)
                cv2.imshow('author', bAuth)
                cv2.waitKey(0)
            ################# Highlight phrase ################################
            binC3 = hBinary.copy()
            (contours, _) = cv2.findContours(~binC3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            heights = [cv2.boundingRect(contour)[3] for contour in contours]  # collecting heights of each contour
            f = 0

            Hlist = []
            if (len(heights) > 0):
                avgheight = math.ceil(sum(heights) / len(heights))
                l = len(colList)
                for i in range(l):
                    lineImg = colList[i]
                    lc = lineImg.copy()
                    blur = cv2.GaussianBlur(lc, (3, 3), 0)
                    canny = cv2.Canny(blur, 120, 255, 1)
                    dil = cv2.dilate(canny, np.ones((1, 10)), 2)
                    dilInv = cv2.bitwise_not(dil)
                    highLine = dilInv.copy()

                    # avg char height
                    charLine = dilInv.copy()
                    (contours2, _) = cv2.findContours(~charLine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    heights2 = [cv2.boundingRect(contour2)[3] for contour2 in
                                contours2]  # collecting heights of each contour

                    if (len(heights2) > 0):
                        avgheight2 = math.floor(sum(heights2) / len(heights2))

                        # calculate white blob
                        highLine[highLine == 0] = 1  # Convert black spots to ones
                        highLine[highLine == 255] = 0  # Convert white spots to zeros
                        hHist = np.sum(highLine, axis=0)  # vertical projection

                        # print("hist",authHist)
                        inBlock = False
                        blob = 0
                        bloSize = {}
                        lenImg = len(hHist)
                        for i in range(lenImg):
                            if (hHist[i] == 0):
                                blob = blob + 1

                            elif (hHist[i] > 0):
                                bloSize[i] = blob
                                blob = 0

                        end = -1
                        for c in reversed(bloSize):
                            if (bloSize[c] == 0):  # black
                                end = c
                                break

                        start = 0
                        for key in bloSize:
                            if (bloSize[key] > 0):  # white
                                start = key
                                break

                        threshEnd = math.floor(lenImg * 0.97)
                        threshStart = math.ceil(lenImg * 0.03)
                        halfs = math.floor(lenImg * 0.5)
                        halfe = math.floor(lenImg * 0.5)

                        if ((end < threshEnd and end > halfe) and (
                                start > threshStart and start < halfs) and avgheight2 > 1.3 * avgheight):
                            resp[1] = 1
                            Hlist.append(lineImg)


            else:
                continue
    for g in range(len(Hlist)):
        cv2.imwrite(os.path.join(pathHl, str(g) + '.jpg'), Hlist[g])
        cv2.imshow("Highlight", Hlist[g])
        cv2.waitKey(0)

    z = len(colList)
    for i in range(z):
        cv2.imwrite(os.path.join(path, str(i) + '.jpg'), colList[i])

    return resp