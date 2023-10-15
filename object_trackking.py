import cv2

def track_object(video_file):
    # Haar Cascade sınıflandırıcısını yükleyin
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Videoyu açın
    video = cv2.VideoCapture(video_file)

    # Videonun boyutlarını alın
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Videoyu çerçeve çerçeve işleyin
    while True:
        # Bir sonraki çerçeveyi alın
        success, frame = video.read()

        # Çerçeveyi gri tonlamaya dönüştürün
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nesneyi tespit edin
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Nesne tespit edilirse, konumunu takip edin
        if len(faces) > 0:
            # Nesnenin merkez koordinatlarını bulun
            (x, y, w, h) = faces[0]
            center = (int(x + w / 2), int(y + h / 2))

            # Nesnenin konumunu çerçeveye çizin
            cv2.circle(frame, center, 20, (0, 255, 0), 2)

        # Çerçeveyi gösterin
        cv2.imshow("Takip", frame)

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
    video_file = "test.mp4"

    # Nesneyi takip edin
    track_object(video_file)
