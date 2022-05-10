import os
import cv2
from FYP_Final.preprocess import preprocessing
from FYP_Final.rawColumns import rawColumns
from FYP_Final.titleSeg import titleCont
from FYP_Final.subTitle import subTitle
from FYP_Final.specialPara import finalCols
from FYP_Final.authorPullQuote import authHighlight
from FYP_Final.sparaLines import spLines
from FYP_Final.words import words
from FYP_Final.contentWords import contWords
from FYP_Final.characterSegmentation import charSegment

path = 'C:/Users/Shashi/Documents/FYP/FYP_Final/DataSet/FArticles/'

# list = os.listdir(path) # dir is your directory path
# number_files = len(list)

i = 2
num = str(i)
type = ".jpg"
im = path + num + type
image = cv2.imread(im)
pathOutput = 'C:/Users/Shashi/Documents/FYP/FYP_Final/Outputs/' + num
os.mkdir(pathOutput)
pathchar = 'C:/Users/Shashi/Documents/FYP/FYP_Final/CharacterElements/' + num
os.mkdir(pathchar)
prepImg = preprocessing(image, num)  # prepocess images
rawColPath = rawColumns(prepImg, num)  # raw column segmentation
titledata = titleCont(num)  # segment the title, content
spara = finalCols(num, titledata[0], titledata[1], titledata[2])  # special para and columns into seperate images
subT = subTitle(num)  # segment the title image into subtitle and main title upto its lines
authHigh = authHighlight(num)  # segment author name and pull quote upto lines

# words segmentation

spParaL = False
subTL = False
authL = False
highL = False
if (spara == True):
    spLines(num)
    spParaL = True
    words(num, 'spParaLines')

if (subT == True):
    subTL = True
    words(num, 'subTitleLines')
if (authHigh[0] == 1):
    authL = True
    words(num, 'authLines')
if (authHigh[1] == 1):
    highL = True
    words(num, 'highlights')

titleW = words(num, 'titleLines')
contW = contWords(num)

# character segmentation
charSegment(num, 'columnwords', 'columns')
charSegment(num, 'titleLineswords', 'title')
if (subTL):
    charSegment(num, 'subTitleLineswords', 'subtitle')
if (highL):
    charSegment(num, 'highlightswords', 'highlights')
if (authL):
    charSegment(num, 'authLineswords', 'reference')
if (spParaL):
    charSegment(num, 'spParaLineswords', 'multiColumnPara')





