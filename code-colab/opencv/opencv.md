# Computer Vision - OpenCV Python

## Table of Contents

- [Computer Vision - OpenCV Python](#computer-vision---opencv-python)
  - [Table of Contents](#table-of-contents)
    - [Import Module](#import-module)
    - [Reading Images](#reading-images)
    - [Reading Videos](#reading-videos)
    - [Resizing and Rescaling](#resizing-and-rescaling)

### Import Module

```py
import cv2 as cv
```

### Reading Images

```py
# read images and store in a variable
img = cv.imread('./../images/1.png')

# show images
cv.imshow('window_name', img) # img is a matrix

# waitKey(0)
# infinitely waiting for keyboard action
cv.waitKey(0)
```

### Reading Videos

```py
# from camera
# capture = cv.VideoCapture(0) # camera 1
# capture = cv.VideoCapture(1) # camera 2
# capture = cv.VideoCapture(3 # camera 2

# from file video 
capture = cv.VideoCapture('project.mp4')

while True:
    flag, frame = capture.read()
    cv.imshow('window', frame)

    if cv.waitKey(20) and 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
```

### Resizing and Rescaling