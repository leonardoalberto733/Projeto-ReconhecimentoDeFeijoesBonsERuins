import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Lista das imagens (3 bons, 3 ruins)
#reservei a img4 (bom) e a img8 (ruim) para teste
feijoes = [
    'Feijoes-editados/img1.jpg', 'Feijoes-editados/img2.jpg', 'Feijoes-editados/img3.jpg', 'Feijoes-editados/img4.jpg'
    'Feijoes-editados/img5.jpg', 'Feijoes-editados/img6.jpg', 'Feijoes-editados/img7.jpg','Feijoes-editados/img8.jpg'
]

caracteristicas = []  # Lista para acumular todas as características

global_id = 0  # ID global para feijões

for x in range(8):
    print(f'processando a imagem {x}...')
    if x <= 3:
        ehBom = 1
    else:
        ehBom = 0

    img = cv2.imread(feijoes[x])
    if img is None:
        print(f"Erro ao carregar imagem: {feijoes[x]}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (17, 17), 0)
    edges = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contorno in enumerate(contornos):
        perimetro = cv2.arcLength(contorno, True)
        if 200 <= perimetro <= 750:
            mascara = np.zeros_like(gray)
            cv2.drawContours(mascara, [contorno], -1, 255, -1)

            feijao_segmentado = cv2.bitwise_and(img, img, mask=mascara)
            pixels = feijao_segmentado[mascara == 255]
            if len(pixels) == 0:
                continue

            kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
            kmeans.fit(pixels)
            cor_dominante = kmeans.cluster_centers_[0]  # B, G, R

            hsv = cv2.cvtColor(feijao_segmentado, cv2.COLOR_BGR2HSV)
            saturacao = hsv[:, :, 1][mascara == 255].mean()

            feijao_gray = cv2.cvtColor(feijao_segmentado, cv2.COLOR_BGR2GRAY)
            pixels_gray = feijao_gray[mascara == 255]
            claros = np.sum(pixels_gray > 200)
            escuros = np.sum(pixels_gray < 50)

            textura = pixels_gray.std()

            area = cv2.contourArea(contorno)
            circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro != 0 else 0

            caracteristicas.append([
                global_id,
                int(cor_dominante[2]),  # R
                int(cor_dominante[1]),  # G
                int(cor_dominante[0]),  # B
                saturacao,
                claros,
                escuros,
                textura,
                circularidade,
                ehBom
            ])

            global_id += 1

#criando arquivo CSV
colunas = ['ID', 'R', 'G', 'B', 'Saturação', 'Claros', 'Escuros', 'Textura', 'Circularidade', 'ehBom']
df = pd.DataFrame(caracteristicas, columns=colunas)
df.to_csv('feijoes_caracteristicas.csv', index=False)
print("Arquivo 'feijoes_caracteristicas.csv' salvo com sucesso!")

