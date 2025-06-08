import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import joblib


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    heigth = int(frame.shape[0] * scale)
    dimensions = (width, heigth)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Lista das imagens (3 bons, 3 ruins)
#reservei a img4 (bom) e a img8 (ruim) para teste
caracteristicas = []  # Lista para acumular todas as características

print('processando a imagem...')

ehBom = 2 #!!!!!!!!!!!!!
img = cv2.imread('Feijoes-editados/img5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 0)
edges = cv2.Canny(blur, 50, 150)

contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contornos_validos_idx = []
for i, contorno in enumerate(contornos):
    perimetro = cv2.arcLength(contorno, True)
    if 200 <= perimetro <= 750:
        contornos_validos_idx.append(i) #salva o índice dos contornos válidos

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


#criando arquivo CSV
colunas = ['ID', 'R', 'G', 'B', 'Saturação', 'Claros', 'Escuros', 'Textura', 'Circularidade', 'ehBom']
df = pd.DataFrame(caracteristicas, columns=colunas)
df.to_csv('feijoesTeste.csv', index=False)
print("Arquivo 'feijoes_caracteristicas.csv' salvo com sucesso!")



#carrega o modelo
mlp = joblib.load('modelo_feijoes.pkl')

# Carregar os dados do novo CSV
novos_df = pd.read_csv('feijoesTeste.csv')

# Remover colunas ID e ehBom (se existirem)
novos_X = novos_df.drop(columns=['ID', 'ehBom']).values

# Fazer a previsão com a rede treinada
predicoes = mlp.predict(novos_X)


for i, pred in enumerate(predicoes):
    idx_real = contornos_validos_idx[i]
    classe = 'Bom' if pred == 1 else 'Ruim'
    print(f"Contorno {idx_real} é classificado como: {classe}")


for j, pred in enumerate(predicoes):
    idx_original = contornos_validos_idx[j]
    cor = (0, 255, 0) if pred == 1 else (0, 0, 255)  # Verde se bom, vermelho se ruim
    cv2.drawContours(img, contornos, idx_original, cor, 5)

# Mostrar ou salvar a imagem resultante
#img = rescaleFrame(img, 0.20)
cv2.imwrite('imagens-geradas/feijoesClassificados.jpg', img)
cv2.imshow('Classificação dos Feijões', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

