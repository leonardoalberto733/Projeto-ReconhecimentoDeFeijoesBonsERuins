import cv2
import numpy as np
from sklearn.cluster import KMeans
import time

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimensions = (width, heigth)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Carrega imagem
img = cv2.imread('Feijoes-editados/img8.jpg')

# Pré-processamento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 0)
edges = cv2.Canny(blur, 50, 150)

# Contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cria uma cópia da imagem original para desenhar os contornos e números
img_contornos = img.copy()

# Desenha os contornos na imagem (verde, espessura 2) e escreve o número do contorno
for i, contorno in enumerate(contornos):
    perimetro = cv2.arcLength(contorno, True)
    if 200 <= perimetro <= 750:
        cv2.drawContours(img_contornos, [contorno], -1, (0, 255, 0), 2)
        
        # Calcula centro do contorno para posicionar o texto
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Se área 0, posiciona o texto no primeiro ponto do contorno
            cx, cy = contorno[0][0]

        # Escreve o número do contorno perto do centro (pode ajustar deslocamento)
        cv2.putText(img_contornos, str(i), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0), 4)
    else:
        print(cv2.arcLength(contorno, True))

# Redimensiona para exibição
img_contornos_resized = rescaleFrame(img_contornos, 0.20)
cv2.imshow('Contornos com IDs', img_contornos_resized)

# Criar e mostrar a máscara só dos contornos válidos
mascara = np.zeros_like(gray)
for contorno in contornos:
    area = cv2.contourArea(contorno)
    if 0 <= area <= 20000:
        cv2.drawContours(mascara, [contorno], -1, 255, 2)

mascara2 = rescaleFrame(mascara, 0.20)
cv2.imshow('Mascara contornos', mascara2)
img = rescaleFrame(img, 0.20)
blur = rescaleFrame(blur, 0.20)
cv2.imshow('original', img)
cv2.imshow('blur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
