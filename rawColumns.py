import math
import os
import cv2
import numpy as np
import itertools
import operator
from matplotlib import pyplot as plt
from pythonRLSA import rlsa


def rawColumns(img,num):
    print("raw columns segmentation..")
    image = img
    (thresh, binary) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #rlsa vertical
    b1 = binary.copy()
    vt = math.ceil(image.shape[0] / 20)
    rl = rlsa.rlsa(b1, False, True, vt)
    #cv2.imwrite('rl.jpg',rl)

    height = image.shape[0]
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi  # angular resolution in radians of the Hough grid
    threshold = math.ceil(height / 3)  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = math.ceil(height / 2)  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    line_image = np.ones(image.shape[:2], dtype="uint8") * 255  # creating a blank to draw lines on

    # Vertical lines
    columnLines = image.copy()
    vertical_lines = cv2.HoughLinesP(rl, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    if vertical_lines is not None:
        for line in vertical_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 0, 2)
                cv2.line(columnLines, (x1, y1), (x2, y2), 0, 2)
    #cv2.imwrite('li.jpg',line_image)
    # project through lines and save :verticle projection of line image
    projCopy = line_image.copy()
    projCopy[projCopy == 0] = 1
    projCopy[projCopy == 255] = 0
    vertical_projection = np.sum(projCopy, axis=0);
    max_height = np.max(vertical_projection)  # max col height in line image
    row_traverse = math.ceil(max_height * 0.60)  # row to traverse throgh to segment the cols
    rowValues = line_image[row_traverse, :]

    width = image.shape[1]
    grops = [[i for i, value in it] for key, it in itertools.groupby(enumerate(rowValues), key=operator.itemgetter(1))
             if key != 255]


    if(len(grops)==0):
        print("please add clear image")
    else:
        # check if some columns are empty: if empty add start and end indexes else add medians
        eff_width = math.ceil(width / 5)
        medians = []
        for w in range(0, len(grops)):
            if (len(grops[w]) > eff_width):
                medians.append(grops[w][0])
                medians.append(grops[w][-1])
            else:
                medians.append(math.floor(np.median(grops[w])))

        # clean medians array to remove points that are too close to each
        minCol = math.floor(width * 0.1)
        cleanMed = []
        if (len(medians) == 1):
            cleanMed.append(medians[0])
        else:
            for i in range(len(medians) - 1):
                if (medians[i + 1] - medians[i] > minCol):
                    cleanMed.append(medians[i])


            if (len(medians) > 1 and medians[-1] - medians[-2] > minCol):
                cleanMed.append(medians[-1])

        gap = 0
        count = 1
        # check last column is included
        for l in range(0, len(cleanMed) - 1):
            gap = gap + cleanMed[l + 1] - cleanMed[l]
            count = count + 1

        if (count == 1 or count == 0):
            avg_col_gap = image.shape[1]
        else:
            avg_col_gap = math.ceil(gap / count - 1)

        if (len(cleanMed) > 0 and width - cleanMed[-1] > math.ceil(0.75 * avg_col_gap)):
            cleanMed.append(width)

        # draw lines and seperate the columns
        roilist = []
        startIndex = 0
        endIndex = 0
        k = 0
        for i in cleanMed:
            if (i < width * 0.1 and cleanMed.index(i) == 0):
                continue

            else:
                endIndex = i
                roiImage = []
                roiImage = np.array(roiImage, dtype="object")
                roiImage = binary[0: height, startIndex: endIndex]
                roilist.insert(k, roiImage)
                k = k + 1
                startIndex = i

        # save columns as images
        l = len(roilist)
        path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num + '/rawCols'
        os.mkdir(path)
        for i in range(l):
            cv2.imwrite(os.path.join(path, str(i) + '.jpg'), roilist[i])
            plt.subplot(3, 3, i + 1), plt.imshow(roilist[i], 'gray')
            plt.xticks([]), plt.yticks([])

        cv2.namedWindow('potentialcolumnMargins', cv2.WINDOW_NORMAL)
        cv2.imshow('potentialcolumnMargins', columnLines)
        cv2.waitKey(0)
        plt.show()
        return path