import cv2
import numpy as np

# Carrega a imagem
img = cv2.imread('Feijoes-editados/img1.jpg')

# Pré-processamento
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# Encontra contornos
contornos, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Para cada contorno encontrado (possível feijão)
for i, contorno in enumerate(contornos):
    # Cria uma máscara preta com o mesmo tamanho da imagem
    mascara = np.zeros_like(gray)
    
    # Desenha o contorno preenchido na máscara
    cv2.drawContours(mascara, [contorno], -1, 255, -1)
    
    # Aplica a máscara para isolar o feijão
    feijao_segmentado = cv2.bitwise_and(img, img, mask=mascara)
    
    # Mostra o resultado
    cv2.imshow(f'Feijao {i}', feijao_segmentado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

