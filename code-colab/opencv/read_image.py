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
cv2.waitKey(0) # 0 means infinitely waiting
