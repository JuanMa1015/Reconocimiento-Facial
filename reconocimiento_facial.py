"""Reconocimiento facial con OpenCV (Haar + LBPH), basado en el tutorial de OMES.

Uso:
  python reconocimiento_facial.py capturar Juan   # 300 fotos de tu rostro
  python reconocimiento_facial.py capturar Otros  # (opcional) rostros de otras personas
  python reconocimiento_facial.py entrenar        # genera modeloLBPH.xml
  python reconocimiento_facial.py reconocer       # cámara en vivo, 'q' o ESC para salir
"""
import os
import sys

import cv2
import numpy as np

NOMBRE_COMPLETO = "Juan Manuel Londono Gonzalez"
PERSONA = "Juan"  # carpeta en Data/ que corresponde a ti
BASE = os.path.dirname(os.path.abspath(__file__))  # Data y modelo siempre junto al script
DATA = os.path.join(BASE, "Data")
MODELO = os.path.join(BASE, "modeloLBPH.xml")
UMBRAL = 70  # distancia LBPH: menor = más parecido. Bájalo si reconoce a otros como tú.
MUESTRAS = 300

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def abrir_camara():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("Error: no se pudo abrir la cámara.")
    return cap


def rostros(gris):
    # Solo devuelve rostros: objetos y fondo no pasan el detector Haar
    return detector.detectMultiScale(gris, 1.3, 5, minSize=(80, 80))


def capturar(persona):
    carpeta = os.path.join(DATA, persona)
    os.makedirs(carpeta, exist_ok=True)
    count = len(os.listdir(carpeta))
    meta = count + MUESTRAS
    cap = abrir_camara()
    while count < meta:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in rostros(gris):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = cv2.resize(gris[y:y + h, x:x + w], (150, 150))
            cv2.imwrite(os.path.join(carpeta, f"rostro_{count}.jpg"), rostro)
            count += 1
        cv2.putText(frame, f"{count}/{meta}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capturando rostros", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{count} imágenes en {carpeta}")


def entrenar():
    personas = sorted(os.listdir(DATA))
    caras, etiquetas = [], []
    for label, persona in enumerate(personas):
        carpeta = os.path.join(DATA, persona)
        for archivo in os.listdir(carpeta):
            caras.append(cv2.imread(os.path.join(carpeta, archivo), cv2.IMREAD_GRAYSCALE))
            etiquetas.append(label)
        print(f"{persona}: {etiquetas.count(label)} imágenes (etiqueta {label})")
    modelo = cv2.face.LBPHFaceRecognizer_create()
    modelo.train(caras, np.array(etiquetas))
    modelo.write(MODELO)
    print(f"Modelo guardado en {MODELO}")


def es_persona(label, distancia, personas):
    return personas[label] == PERSONA and distancia < UMBRAL


def reconocer():
    personas = sorted(os.listdir(DATA))
    modelo = cv2.face.LBPHFaceRecognizer_create()
    modelo.read(MODELO)
    cap = abrir_camara()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in rostros(gris):
            rostro = cv2.resize(gris[y:y + h, x:x + w], (150, 150))
            label, distancia = modelo.predict(rostro)
            if es_persona(label, distancia, personas):
                texto, color = NOMBRE_COMPLETO, (0, 255, 0)
            else:
                texto, color = "Desconocido", (0, 0, 255)
            cv2.putText(frame, f"{distancia:.0f}", (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow("Reconocimiento facial", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "capturar":
        capturar(sys.argv[2] if len(sys.argv) > 2 else PERSONA)
    elif sys.argv[1] == "entrenar":
        entrenar()
    elif sys.argv[1] == "reconocer":
        reconocer()
    elif sys.argv[1] == "test":
        p = ["Juan", "Otros"]
        assert es_persona(0, 40, p)
        assert not es_persona(0, 90, p)  # se parece poco: desconocido
        assert not es_persona(1, 20, p)  # es otra persona entrenada
        print("ok")
    else:
        sys.exit(__doc__)
