import cv2

cap = cv2.VideoCapture(0)
bgs = cv2.bgsegm.createBackgroundSubtractorLSBP()

while(cap.isOpened()):
    ret, frame = cap.read()
    resize_img = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
    mask = bgs.apply(resize_img)
    bg = bgs.getBackgroundImage()
    cv2.imshow('mask', mask)
    cv2.imshow('bg', bg)
    if cv2.waitKey(1) != -1:
        break

cap.release()
cv2.destroyAllWindows()