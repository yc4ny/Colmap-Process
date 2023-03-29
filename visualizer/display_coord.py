import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: ({}, {})".format(x, y))

image = cv2.imread("preprocessed/region_left/00031.jpg")  # replace with your image path
cv2.imshow("Image", image)

cv2.setMouseCallback("Image", click_event)
cv2.resizeWindow("Image", 600, 400)  # set window size

cv2.waitKey(0)
cv2.destroyAllWindows()