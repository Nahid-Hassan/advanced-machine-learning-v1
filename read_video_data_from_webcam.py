import cv2

# create webcam object
capture = cv2.VideoCapture(0)
# set some parameter
# set width, which id number is 3
capture.set(3, 240)
# set height, which id number is 4
capture.set(4, 480)
# set brightness of webcamp, which id number is 10
capture.set(10, 100)


while True:
    status, frame = capture.read()

    cv2.imshow('Webcam 0', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break