import os
import cv2
import numpy as np
import math


def finalCols(num, numCols, xVal,point):
    print("final columns segmenting...")
    imNum = str(num)
    contentImage = cv2.imread('C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum + '/Content.jpg', cv2.IMREAD_GRAYSCALE)
    cImg = contentImage.copy()
    numColumn = numCols
    sp = False
    if (numColumn > 1):
        bin_inv = cv2.bitwise_not(contentImage)
        cannyCont = cv2.Canny(bin_inv, 50, 150)
        kernel = np.ones((9, 9))
        imgForHorizontal_Dilate = cv2.dilate(cannyCont, kernel, 1)
        maskContent = np.ones(contentImage.shape[:2], dtype="uint8") * 255
        (contours2, _) = cv2.findContours(imgForHorizontal_Dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            [x, y, w, h] = cv2.boundingRect(contour)
            cv2.rectangle(maskContent, (x, y), (x + w, y + h), 0, -1)
        contHist = maskContent.copy()
        contHist[contHist == 0] = 1  # Convert black spots to ones
        contHist[contHist == 255] = 0

        sum = 0
        blCntArray = []

        for bl in range(0, len(xVal) - 1):
            for contHeight in range(contHist.shape[0]):
                sum = sum + contHist[contHeight, xVal[bl]]
            blCntArray.append(sum)
            sum = 0

        flag = True
        fcount = 0
        # find margins
        heightC = contentImage.shape[0]
        blThresh = math.floor(heightC * 0.01)
        for v in range(1, len(blCntArray)):
            if (blCntArray[v] > blThresh):
                paraWidth = xVal[v + 1] - xVal[0]
                paraImg = contHist[0:heightC, 0:paraWidth]
                pcImg = contentImage[0:heightC, 0:paraWidth]
                fcount = fcount + 1
            else:
                flag = False

        if (fcount > 0 and blCntArray[1] > blThresh):
            horizontal_projection = np.sum(paraImg, axis=1)  # paraImg histogram
            blackThresh = math.floor(paraWidth * 0.05)

            margins = []
            for a in range(len(horizontal_projection)):
                if (horizontal_projection[a] < blackThresh):
                    margins.append(1)
                else:
                    margins.append(0)

            arr = np.asarray([0] + margins + [0])  # finding indexes(last element of 1s group)
            arr_diff = np.diff(arr)  # finding borders between 0 and 1
            last = np.where(arr_diff == -1)[0] - 1  # finding index values of first and last elements of each group of 1's

            if (len(last) > 1):
                sp = True
                specialPara = contentImage[last[0]:last[1], 0:paraWidth]
                cImg[last[0]:last[1], 0:paraWidth] = 255
                # save special para
                pathPara = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum + '/sp'
                os.mkdir(pathPara)
                cv2.imwrite(os.path.join(pathPara, 'specialPara.jpg'), specialPara)

                cv2.namedWindow('multi coulmn para', cv2.WINDOW_NORMAL)
                cv2.imshow("multi coulmn para", specialPara)
                cv2.waitKey(0)

        else:
            print("No multi column para")
    else:
        print("No multi column para: Image has only one column")

    #######save columns into seperate images
    colList = []
    for v in range(0, len(xVal) - 1):
        colImg = []
        colImg = cImg[point[v]:, xVal[v]:xVal[v + 1]]
        colList.insert(v, colImg)

    l = len(colList)
    pathCol = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + imNum + '/columns/'
    # if (not pathPara):
    os.mkdir(pathCol)
    for i in range(l):
        cv2.imwrite(os.path.join(pathCol, str(i) + '.jpg'), colList[i])

    return sp