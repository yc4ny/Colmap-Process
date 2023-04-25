import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        height, width, channels = param.shape
        y = height - y  # shift y-coordinate to bottom-left origin
        print("Coordinates: ({}, {})".format(x, y))

image = cv2.imread("preprocessed/undistorted_right/right_00165.jpg")  # replace with your image path
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 600, 400)  # set window size
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event, param=image)

cv2.waitKey(0)
cv2.destroyAllWindows()
