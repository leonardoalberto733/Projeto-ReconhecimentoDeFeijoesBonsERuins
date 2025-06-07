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
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lista para armazenar [id, cor_dominante_RGB]
caracteristicas = []

for i, contorno in enumerate(contornos):
    # Cria máscara do feijão
    mascara = np.zeros_like(gray)
    cv2.drawContours(mascara, [contorno], -1, 255, -1)

    # Aplica máscara para extrair pixels do feijão
    feijao_segmentado = cv2.bitwise_and(img, img, mask=mascara)

    # Extrai apenas os pixels dentro da máscara
    pixels = feijao_segmentado[mascara == 255]
    
    if len(pixels) == 0:
        continue  # pula se não encontrou pixels

    # Aplica KMeans com 1 cluster (cor dominante)
    kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
    kmeans.fit(pixels)
    cor_dominante = kmeans.cluster_centers_[0]  # [B, G, R]

    # Armazena [id, R, G, B]
    caracteristicas.append([i, int(cor_dominante[2]), int(cor_dominante[1]), int(cor_dominante[0])])  # RGB

    cv2.imshow('feijao_segmentado {i}', rescaleFrame(feijao_segmentado, 0.20))
    cv2.waitKey(0)
    if i > 100:
        break

# Converte para array NumPy
caracteristicas = np.array(caracteristicas)

# Mostra o resultado
print("Matriz de características (ID, R, G, B):")
print(caracteristicas)