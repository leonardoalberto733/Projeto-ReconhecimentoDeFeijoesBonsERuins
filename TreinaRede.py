import pandas as pd
import joblib
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

#salva o modelo
joblib.dump(mlp, 'modelo_feijoes.pkl')

# Passo 5: avaliar no conjunto de teste
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no teste: {accuracy*100:.2f}%\n")

print("Relatório de classificação:")
print(classification_report(y_test, y_pred))










