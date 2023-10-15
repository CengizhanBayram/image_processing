import cv2

def detect_humans_in_video(video_file):
    # Haar Cascade sınıflandırıcısını yükleyin
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Videoyu açın
    video = cv2.VideoCapture(video_file)

    # Videonun boyutlarını alın
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Videoyu çerçeve çerçeve işleyin
    while video.isOpened():
        # Bir sonraki çerçeveyi alın
        success, frame = video.read()

        # Çerçeveyi gri tonlamaya dönüştürün
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # İnsanları tespit edin
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # İnsanları çerçeveye çizin
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Çerçeveyi gösterin
        cv2.imshow("İnsanlar", frame)

        # Bir tuşa basılıncaya kadar bekleyin
        key = cv2.waitKey(1)

        # Çıkış için ESC tuşuna basın
        if key == 27:
            break

    # Videoyu kapatın
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Video dosyasını belirtin
    video_file = "my_video.mp4"

    # Videodan insanları tespit edin
    detect_humans_in_video(video_file)
