import cv2

video = r"C:\Users\Tamro\Downloads\Teste ET.mov"

cap = cv2.VideoCapture(video)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()