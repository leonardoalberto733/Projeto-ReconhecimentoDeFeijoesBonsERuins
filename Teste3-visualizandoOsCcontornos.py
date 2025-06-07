import cv2
import numpy as np
from sklearn.cluster import KMeans

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)

    dimensions = (width, heigth)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Carrega imagem
img = cv2.imread('Feijoes-editados/img1.jpg')

# Pré-processamento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blur, 50, 150)

# Contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cria uma cópia da imagem original para desenhar os contornos
img_contornos = img.copy()

# Desenha os contornos na imagem (em verde com espessura 2)
cv2.drawContours(img_contornos, contornos, -1, (0, 255, 0), 2)

# Exibe a imagem com os contornos
img_contornos = rescaleFrame(img_contornos, 0.20)

cv2.imshow('Contornos', img_contornos)



mascara = np.zeros_like(gray)
img_contornos = img.copy()

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if 700 <= area <= 10000:
        cv2.drawContours(mascara, [contorno], -1, 255, 2)


mascara = rescaleFrame(mascara, 0.20)
cv2.imshow('contornos', mascara)


cv2.waitKey(0)
cv2.destroyAllWindows()
