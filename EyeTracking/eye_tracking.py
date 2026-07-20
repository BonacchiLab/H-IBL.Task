import cv2
import mediapipe as mp
import csv
import os

# ==========================
# CONFIGURAÇÃO
# ==========================

VIDEO_PATH = r"C:\Users\Tamro\Downloads\Teste ET (5).mov"
OUTPUT_FOLDER = "resultados"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

csv_path = os.path.join(OUTPUT_FOLDER, video_name + "_resultados.csv")
video_output = os.path.join(OUTPUT_FOLDER, video_name + "_anotado.mp4")

# ==========================
# ABRIR VÍDEO
# ==========================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(video_output, fourcc, fps, (largura, altura))

# ==========================
# CSV
# ==========================

csv_file = open(csv_path, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Tempo (s)", "Olhar"])

# ==========================
# MEDIAPIPE TASKS (API NOVA)
# ==========================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker_v2_with_blendshapes.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

frame_number = 0

print("Processamento iniciado...")

# ==========================
# CICLO PRINCIPAL
# ==========================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    tempo = frame_number / fps

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect(mp_image)

    olhar = "CENTRO"

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        # ==========================
        # ÍRIS ESQUERDA (468–471)
        # ==========================
        iris_left_x = (
            landmarks[468].x +
            landmarks[469].x +
            landmarks[470].x +
            landmarks[471].x
        ) / 4

        left_outer = landmarks[33].x
        left_inner = landmarks[133].x

        ratio_left = (iris_left_x - left_outer) / (left_inner - left_outer + 1e-6)

        # ==========================
        # ÍRIS DIREITA (473–476)
        # ==========================
        iris_right_x = (
            landmarks[473].x +
            landmarks[474].x +
            landmarks[475].x +
            landmarks[476].x
        ) / 4

        right_outer = landmarks[362].x
        right_inner = landmarks[263].x

        ratio_right = (iris_right_x - right_outer) / (right_inner - right_outer + 1e-6)

        # ==========================
        # MÉDIA DOS DOIS OLHOS
        # ==========================
        ratio = (ratio_left + ratio_right) / 2

        # ==========================
        # DEBUG NO TERMINAL
        # ==========================
        print(f"LEFT={ratio_left:.3f} | RIGHT={ratio_right:.3f} | AVG={ratio:.3f}")

        # ==========================
        # AUTO-CALIBRAÇÃO
        # ==========================
        left_threshold = 0.45
        right_threshold = 0.55

        if ratio < 0.48:
            left_threshold = 0.50
        if ratio > 0.52:
            right_threshold = 0.50

        # ==========================
        # CLASSIFICAÇÃO DO OLHAR (INVERTIDO)
        # ==========================
        if ratio < left_threshold:
            olhar = "DIREITA"   # invertido
        elif ratio > right_threshold:
            olhar = "ESQUERDA"  # invertido
        else:
            olhar = "CENTRO"

        # ==========================
        # DESENHAR ÍRIS DOS DOIS OLHOS
        # ==========================
        for idx in [468, 469, 470, 471, 473, 474, 475, 476]:
            x = int(landmarks[idx].x * largura)
            y = int(landmarks[idx].y * altura)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ==========================
        # DEBUG VISUAL NO VÍDEO
        # ==========================
        cv2.putText(frame, f"Ratio AVG: {ratio:.3f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"L={ratio_left:.3f} R={ratio_right:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"TH: {left_threshold:.2f} / {right_threshold:.2f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

    # Texto no vídeo
    cv2.putText(frame, f"Olhar: {olhar}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # CSV
    csv_writer.writerow([frame_number, round(tempo, 3), olhar])

    # Vídeo
    writer.write(frame)

    # Mostrar
    cv2.imshow("Eye Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# FECHAR TUDO
# ==========================

csv_file.close()
writer.release()
cap.release()
cv2.destroyAllWindows()

print("\n===================================")
print("Processamento terminado!")
print("CSV guardado em:")
print(csv_path)
print("\nVídeo anotado guardado em:")
print(video_output)
print("===================================")
