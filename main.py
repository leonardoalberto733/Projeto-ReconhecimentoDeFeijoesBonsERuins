import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

#img1-4 são feijões bons e img5-8 são feijões ruins
#então feijoes[0-3] são feijões bons e feijoes[4-7] são feijões ruins
feijoes = ['Feijoes-editados/img1.jpg', 'Feijoes-editados/img2.jpg', 'Feijoes-editados/img3.jpg', 'Feijoes-editados/img4.jpg','Feijoes-editados/img5.jpg', 'Feijoes-editados/img6.jpg', 'Feijoes-editados/img7.jpg', 'Feijoes-editados/img8.jpg']


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimensions = (width, heigth)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


# Configura pandas para mostrar todas as linhas e colunas sem truncar
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


# Carrega imagem
for x in range(8):
    if x <= 3:
        ehBom = 1
    else:
        ehBom = 0

    img = cv2.imread(feijoes[x])

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (17, 17), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Contornos
    contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar características
    caracteristicas = []

    for i, contorno in enumerate(contornos):
        perimetro = cv2.arcLength(contorno, True)
        if 200 <= perimetro <= 750:
            # Cria máscara do feijão
            mascara = np.zeros_like(gray)
            cv2.drawContours(mascara, [contorno], -1, 255, -1)

            # Aplica máscara para extrair pixels do feijão
            feijao_segmentado = cv2.bitwise_and(img, img, mask=mascara)

            # Extrai pixels dentro da máscara
            pixels = feijao_segmentado[mascara == 255]
            
            if len(pixels) == 0:
                continue  # pula se não encontrou pixels

            # KMeans para cor dominante
            kmeans = KMeans(n_clusters=1, random_state=0, n_init=10)
            kmeans.fit(pixels)
            cor_dominante = kmeans.cluster_centers_[0]  # [B, G, R]

            # Saturação
            hsv = cv2.cvtColor(feijao_segmentado, cv2.COLOR_BGR2HSV)
            saturacao = hsv[:, :, 1][mascara == 255].mean()

            # Pixels claros e escuros (em grayscale)
            feijao_gray = cv2.cvtColor(feijao_segmentado, cv2.COLOR_BGR2GRAY)
            pixels_gray = feijao_gray[mascara == 255]
            claros = np.sum(pixels_gray > 200)
            escuros = np.sum(pixels_gray < 50)

            # Textura (desvio padrão do grayscale)
            textura = pixels_gray.std()

            # Circularidade
            area = cv2.contourArea(contorno)
            circularidade = 4 * np.pi * area / (perimetro ** 2) if perimetro != 0 else 0

            # Armazena características: ID, R, G, B, saturação, claros, escuros, textura, circularidade
            caracteristicas.append([
                i,
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


# Converte para DataFrame pandas
#df = pd.DataFrame(caracteristicas, columns=[
#    'ID', 'R', 'G', 'B', 'Saturação', 'Claros', 'Escuros', 'Textura', 'Circularidade'
#])

# Mostra toda a tabela
print("Matriz de características:")
np.set_printoptions(suppress=True)
for linha in caracteristicas:
    # Formata cada elemento da linha
    linha_formatada = []
    for elem in linha:
        if isinstance(elem, float):
            linha_formatada.append(f"{elem:.3f}")  # float com 3 casas decimais
        else:
            linha_formatada.append(str(int(elem)))  # int como string sem prefixo
    print(linha_formatada)

