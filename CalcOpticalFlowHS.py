import cv
import sys
import cv2
import numpy as np

FLOWSKIP = 1

# if len(sys.argv) != 3:
#  sys.exit("Please input two arguments: imagename1 imagename2")

# inputImageFirst = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
# inputImageSecond = cv.LoadImage(sys.argv[2], cv.CV_LOAD_IMAGE_GRAYSCALE)

inputImageFirst = cv.LoadImage('./A/8.0/shuibo_9.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
inputImageSecond = cv.LoadImage('./A/8.0/shuibo_10.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

# desImageHS = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)
# desImageLK = cv.LoadImage(sys.argv[1], cv.CV_LOAD_IMAGE_COLOR)


desImageHS = cv.LoadImage('./A/8.0/shuibo_9.jpg', cv.CV_LOAD_IMAGE_COLOR)
desImageLK = cv.LoadImage('./A/8.0/shuibo_9.jpg', cv.CV_LOAD_IMAGE_COLOR)

cols = inputImageFirst.width
rows = inputImageFirst.height

velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
cv.SetZero(velx)
cv.SetZero(vely)

cv.CalcOpticalFlowHS(inputImageFirst, inputImageSecond, False, velx, vely, 100.0,
                     (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))
f=open('./A/8.0/shuibo_8_HS(x1,y1,x1,y2).txt','w')
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

cv.SetZero(velx)
cv.SetZero(vely)

#包含了实现CalcOpticalFlowLK的方法
cv.CalcOpticalFlowLK(inputImageFirst, inputImageSecond, (15, 15), velx, vely)

for i in range(0, cols, FLOWSKIP):
    for j in range(0, rows, FLOWSKIP):
        dx = int(cv.GetReal2D(velx, j, i))
        dy = int(cv.GetReal2D(vely, j, i))
        cv.Line(desImageLK, (i, j), (i + dx, j + dy), (0, 0, 255), 1, cv.CV_AA, 0)

cv.SaveImage("resultHS.png", desImageHS)
cv.SaveImage("resultLK.png", desImageLK)

cv.NamedWindow("Optical flow HS")
cv.ShowImage("Optical flow HS", desImageHS)

cv.NamedWindow("Optical flow LK")
cv.ShowImage("Optical flow LK", desImageLK)

cv.WaitKey(0)
cv.DestroyAllWindows()