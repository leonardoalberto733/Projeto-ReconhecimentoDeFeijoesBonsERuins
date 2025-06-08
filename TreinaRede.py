import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Passo 1: ler os dados
df = pd.read_csv('feijoes_caracteristicas.csv')

# Passo 2: separar features e label
X = df.drop(columns=['ID', 'ehBom']).values  # características (sem ID e sem a classe)
y = df['ehBom'].values                        # classes (0 ou 1)

# Passo 3: dividir em treino e teste (exemplo 80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: criar e treinar o modelo MLP (rede neural multicamada)
mlp = MLPClassifier(hidden_layer_sizes=(50, ), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Passo 5: avaliar no conjunto de teste
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {accuracy*100:.2f}%\n")

print("Relatório de classificação:")
print(classification_report(y_test, y_pred))









import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Lista das imagens (3 bons, 3 ruins)
#reservei a img4 (bom) e a img8 (ruim) para teste
feijoes = ['Feijoes-editados/img8.jpg']

caracteristicas = []  # Lista para acumular todas as características

global_id = 0  # ID global para feijões

for x in range(1):
    print(f'processando a imagem {x}...')
    if x <= 3:
        ehBom = 0
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
df.to_csv('feijoesTeste.csv', index=False)
print("Arquivo 'feijoes_caracteristicas.csv' salvo com sucesso!")





# Carregar os dados do novo CSV
novos_df = pd.read_csv('feijoesTeste.csv')

# Remover colunas ID e ehBom (se existirem)
novos_X = novos_df.drop(columns=['ID', 'ehBom']).values

# Fazer a previsão com a rede treinada
predicoes = mlp.predict(novos_X)

a = 0
# Exibir os resultados
for i, pred in enumerate(predicoes):
    classe = 'Bom' if pred == 1 else 'Ruim'
    print(f"Feijão {i+1} é classificado como: {classe}")
    if pred != 1:
        a = a+1
print(a)

