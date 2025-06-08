import cv2
import numpy as np

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimensions = (width, heigth)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Carrega imagem
img = cv2.imread('Feijoes-editados/img1.jpg')
img_original = img.copy()

# Pré-processamento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Remove pequenos ruídos
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Área de fundo (sure background)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Área interna dos feijões (sure foreground)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Regiões desconhecidas (entre os feijões)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

# Adiciona 1 a todos os rótulos para que o fundo seja 1 em vez de 0
markers = markers + 1

# Marca como 0 as regiões desconhecidas
markers[unknown == 255] = 0

# Aplica watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [0, 0, 255]  # borda vermelha

# Desenha contornos separados
img_contornos = img_original.copy()
contornos, _ = cv2.findContours((markers > 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, contorno in enumerate(contornos):
    area = cv2.contourArea(contorno)
    if 100 <= area <= 20000:
        cv2.drawContours(img_contornos, [contorno], -1, (0, 255, 0), 2)
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img_contornos, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Mostrar resultado final
cv2.imshow('Separação com Watershed', rescaleFrame(img_contornos, 0.3))
cv2.waitKey(0)
cv2.destroyAllWindows()
