import cv2
import time
from ultralytics import YOLO

# Mülakat notu: Edge AI donanımlarında (ör. Raspberry Pi, Jetson Nano) 
# her zaman en hafif modeller tercih edilir. Bu yüzden 'nano' modelini kullanıyoruz.
model = YOLO("yolov8n.pt")

# Test için standart bir video dosyası. 
# (İnternetten kısa bir trafik mp4 videosu indirip proje dizinine koyabilirsin)
video_path = "traffic.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Hata: Video dosyasi acilamadi. Lutfen 'traffic.mp4' dosyasini kontrol et.")
    exit()

print("YOLOv8 Python Baseline başlatılıyor...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı.")
        break

    # Toplam işlem süresi için kronometreyi başlat
    start_time = time.perf_counter()

    # SADECE ÇIKARIM (INFERENCE) SÜRESİNİ ÖLÇÜYORUZ
    inference_start = time.perf_counter()
    # verbose=False ile terminalin gereksiz loglarla yavaşlamasını engelliyoruz
    results = model(frame, verbose=False) 
    inference_end = time.perf_counter()

    # Sonuçları (kutuları ve etiketleri) çerçevenin üzerine çiz
    annotated_frame = results[0].plot()

    # Toplam süreyi bitir
    end_time = time.perf_counter()

    # Metrikleri milisaniye (ms) cinsinden hesapla
    inference_ms = (inference_end - inference_start) * 1000
    total_ms = (end_time - start_time) * 1000
    fps = 1000 / total_ms if total_ms > 0 else 0

    # Metrikleri ekrana yazdır
    cv2.putText(annotated_frame, f"Inference Latency: {inference_ms:.1f} ms", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Total FPS: {fps:.1f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Python Baseline (Bottleneck)", annotated_frame)

    # 'q' tuşuna basılarak çıkış yapılabilir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()