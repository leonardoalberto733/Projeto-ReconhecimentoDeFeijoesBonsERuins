

import cv2 as cv2
import numpy as np

img = cv2.imread('Feijoes-editados/img1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('feijão borrado', blur)
edges = cv2.Canny(blur, 50, 150)
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mascara = np.zeros_like(gray)
cv2.drawContours(mascara, [contorno], -1, 255, -1)  # Preenche o contorno com branco

# Aplica a máscara na imagem colorida
feijao_segmentado = cv2.bitwise_and(img, img, mask=mascara)

cv2.imshow('feijao segmentado', feijao_segmentado)
cv2.waitKey(0)
