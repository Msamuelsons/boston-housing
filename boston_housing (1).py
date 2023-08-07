# Importando as bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso

# Carregando os dados do conjunto "Boston Housing" em um DataFrame
dados = pd.read_csv('/content/sample_data/housing.csv')

# Visualizando as primeiras linhas do DataFrame para verificar a estrutura dos dados
dados.head()

# Verificando a quantidade de valores nulos em cada coluna do DataFrame
isnull = dados.isnull().sum()
isnull

# Calculando a matriz de correlação entre as colunas do DataFrame
matriz_correlacao = dados.corr()
matriz_correlacao

# Visualizando a matriz de correlação em um gráfico de calor
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Gráfico de Correlação')
plt.show()

# Separando as variáveis independentes (x) e a variável dependente (y) do conjunto de dados
y = dados['MEDV']
x = dados.drop('MEDV', axis=1)

# Definindo os valores de hiperparâmetros para o ElasticNet que serão testados na busca em grid
valores = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000],
           'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

# Criando um objeto KFold para realizar a validação cruzada
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Instanciando o modelo ElasticNet e criando um objeto GridSearchCV para buscar os melhores hiperparâmetros
elasticNet = ElasticNet()
modelo = GridSearchCV(estimator=elasticNet, param_grid=valores, cv=5)
modelo.fit(x, y)

# Realizando a validação cruzada do modelo ElasticNet para avaliar seu desempenho
modelo_elastic = cross_val_score(elasticNet, x, y, cv=kfold)
elastic_modelo = modelo_elastic.mean()

# Instanciando o modelo de Regressão Linear
linearRegression = LinearRegression()

# Realizando a validação cruzada do modelo de Regressão Linear para avaliar seu desempenho
modelo2 = cross_val_score(linearRegression, x, y, cv=kfold)
linear_modelo = modelo2.mean()

# Instanciando o modelo Ridge
ridge = Ridge()

# Realizando a validação cruzada do modelo Ridge para avaliar seu desempenho
modelo3 = cross_val_score(ridge, x, y, cv=kfold)
ridge.fit(x, y)

# Realizando a busca em grid para encontrar o melhor valor de hiperparâmetro alpha para o modelo Ridge
ridge_search_alpha = {'alpha': [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]}
grid_ridge = GridSearchCV(estimator=ridge, param_grid=ridge_search_alpha, cv=5)
grid_ridge.fit(x, y)

# Obtendo o melhor modelo Ridge encontrado na busca em grid e seu score
melhor_alpha = grid_ridge.best_estimator_
melhor_score = grid_ridge.best_score_

# Criando outro modelo Ridge com o melhor valor de hiperparâmetro encontrado
ridge2 = Ridge(alpha=100)

# Realizando a validação cruzada do modelo Ridge com o melhor hiperparâmetro
ridge_modelo = cross_val_score(ridge2, x, y, cv=kfold)
rid_modelo = ridge_modelo.mean()

# Instanciando o modelo Lasso
lasso = Lasso()

# Definindo os valores de hiperparâmetros para o Lasso que serão testados na busca em grid
alpha_lasso = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
max_iter_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
tol_values = [0.01, 0.1, 0.001]
param_lasso = {'alpha': alpha_lasso, 'max_iter': max_iter_values, 'tol': tol_values}

# Criando um objeto GridSearchCV para buscar os melhores hiperparâmetros para o modelo Lasso
grid_lasso = GridSearchCV(estimator=lasso, param_grid=param_lasso, cv=kfold)
grid_lasso.fit(x, y)

# Realizando a validação cruzada do modelo Lasso para avaliar seu desempenho
lasso_modelo = cross_val_score(lasso, x, y, cv=kfold, scoring='r2')
lass_modelo = lasso_modelo.mean()

# Obtendo o melhor modelo Lasso encontrado na busca em grid
melhor_modelo_lasso = grid_lasso.best_estimator_

# Função para retornar o melhor resultado entre os modelos testados
def melhor_resultado(ridge, lasso, linear, elastic):
    return max(ridge, lasso, linear, elastic)

# Chamando a função para encontrar o melhor modelo e seu desempenho
melhor_resultado(rid_modelo, linear_modelo, lass_modelo, elastic_modelo)

# Realizando previsões usando o modelo de Regressão Linear e validação cruzada
predicao = cross_val_predict(linearRegression, x, y, cv=5)

# Exibindo as previsões feitas pelo modelo
print(predicao.tolist())

