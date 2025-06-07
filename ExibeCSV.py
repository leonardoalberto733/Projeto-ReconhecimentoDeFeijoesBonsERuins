import pandas as pd

# Lê o arquivo CSV
df = pd.read_csv('feijoes_caracteristicas.csv')

# Mostra todas as linhas e colunas (opcional, se quiser ver tudo mesmo que seja grande)
pd.set_option('display.max_rows', None)         # Mostra todas as linhas
pd.set_option('display.max_columns', None)      # Mostra todas as colunas
pd.set_option('display.width', None)            # Largura ilimitada para evitar truncamento
pd.set_option('display.max_colwidth', None)     # Colunas não truncadas

# Exibe a tabela
print(df)