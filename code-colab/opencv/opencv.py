import cv2 as cv

def rescaleFrame(frame, scale=.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    dimensions = (width, height)
    return cv.resize9

# read images and store in a variable
img = cv.imread('./../images/1.png')()
# show images
cv.imshow('window_name', img) # img is a matrix

# waitKey(0)
# infinitely waiting for keyboard action
cv.waitKey(0)
