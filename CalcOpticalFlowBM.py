import cv
import sys
import cv2
import numpy as np

FLOWSKIP = 1

# if len(sys.argv) != 3:
#  sys.exit("Please input two arguments: imagename1 imagename2")

# inputImageFirst = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
# inputImageSecond = cv.LoadImage(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)

inputImageFirst = cv.LoadImage('./F/0.5/shuibo_9.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
inputImageSecond = cv.LoadImage('./F/0.5/shuibo_10.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

# desImageHS = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
# desImageLK = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)


desImageHS = cv.LoadImage('./F/0.5/shuibo_9.jpg', cv.CV_LOAD_IMAGE_COLOR)
desImageLK = cv.LoadImage('./F/0.5/shuibo_9.jpg', cv.CV_LOAD_IMAGE_COLOR)

cols = inputImageFirst.width
rows = inputImageFirst.height
print cols,rows

velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
cv.SetZero(velx)
cv.SetZero(vely)

cv.CalcOpticalFlowBM(inputImageFirst, inputImageSecond, (1,1),(1,1),(1,1) ,0,velx, vely)
f=open('./F/0.5/shuibo_05_BM(x1,y1,x1,y2).txt','w')#将光流保存到txt文件中
count=0
for i in range(0, cols, FLOWSKIP):
    for j in range(0, rows, FLOWSKIP):
        dx = int(cv.GetReal2D(velx, j, i))
        dy = int(cv.GetReal2D(vely, j, i))
        cv.Line(desImageHS, (i, j), (i + dx, j + dy), (0, 0, 255), 1, cv.CV_AA, 0)
        f.writelines([ str(i), ' ', str(j), ' ', str(i + dx), ' ', str(j + dy),'\n'])
        # count+=1
        # print count
f.close()

cv.SaveImage("resultHS.png", desImageHS)

cv.NamedWindow("Optical flow HS")
cv.ShowImage("Optical flow HS", desImageHS)


cv.WaitKey(0)
cv.DestroyAllWindows()