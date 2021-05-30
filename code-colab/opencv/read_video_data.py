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
