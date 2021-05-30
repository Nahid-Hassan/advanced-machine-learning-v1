# OpenCV Python

## Table of Contents

- [OpenCV Python](#opencv-python)
  - [Table of Contents](#table-of-contents)
    - [Course Outline](#course-outline)
    - [Introduction to Images](#introduction-to-images)
    - [Installation](#installation)
    - [Read Image](#read-image)
    - [Read Video Data](#read-video-data)

### Course Outline

**Topics**:

1. Read Images, Videos and Webcams
2. Basic Functions
3. Resizing and Cropping
4. Shapes and Text
5. Wrap Perspective
6. Joining Images
7. Color Detection
8. Contour / Shape Detection
9. Face Detection

**Projects**:

1. Virtual Paint
2. Paper Scanner
3. Number Plate Detector

### Introduction to Images

![images](images/1.png)

- VGA = 640 x 480
- HD = 1280 x 720
- FHD = 1920 x 1080
- 4k = 3840 x 2160

**Levels of Images**:

```text
2 levels = 0(black), and 1(white) pixels
6 levels, 16 levels and 256 levels (Gray Scale Images)
```

**Note**: 2^8 = 256(8 bits = 256 levels), where 0 means `black` and 255 is `white`. **GrayScaleImages**.

**Colored Images(RGB)**:

`RGB VGA = 640(width) x 480(height) x 3(channel)`

### Installation

```console
nahid@infoForest: pip3 install opencv-python
```

### Read Image

```py
import cv2

# imread(filename)
# imread() -> i am read
img = cv2.imread('images/1.png')
# imshow('window title', img_matrix)
# imshow() -> i am show
cv2.imshow('3 pixels',img)

# now image is showed but it is immediately disappear. that's why we
# need to call cv2.waitKey()
# cv2.waitKey(delay) # delay in milliseconds
# waitKey() take delay parameter in milliseconds
cv2.waitKey(0) # 0 means infinitely waiting
```

### Read Video Data

```python
import cv2

# read video using VideoCapture(file name)
capture = cv2.VideoCapture('project.mp4')

# VideoCapture(..) read video frame by frame
while True:
    # read method read frame from video
    status, frame = capture.read()
    # show the frame
    cv2.imshow('video', frame)

    # if you press key 'q' it breaks the loop
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
```
