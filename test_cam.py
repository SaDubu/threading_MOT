import cv2

cap = cv2.VideoCapture("/dev/video11", cv2.CAP_V4L2)

if not cap.isOpened():
    print("Camera open failed!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed!")
        break

    cv2.imshow("Video11", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
