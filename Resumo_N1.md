<p align="right">
  <img src="http://meusite.mackenzie.br/rogerio/mackenzie_logo/UPM.2_horizontal_vermelho.jpg" width="30%" align="center"/>
</p>

# Inteligência Artificial - Resumo N1

rogerio.oliveira@mackenzie.br  

<br>

<br>



**1** [**Introdução à Inteligência Artificial: conceito, história e paradigmas**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Introducao.ipynb) 

- IA $\times$ ML $\times$ Deep Learning -> IA é apenas usando lógicamente basicamente, no ML a gente aprende com os dados, a inteligência é extraída a partir dos dados, lendo eles e pegando os padrões, Deep learning é uma parte de ML
- IA Fraca $\times$ IA Forte -> Forte seria a IA ter consciência 
- ML $\times$ Data Science -> Ciência de dados engloba outras tarefas sobre os dados, dentre elas o CRISPM DM (Parte de preparação e conhecimento)
- Como avalio os modelos de ML
- Aplicações ou Tarefas de ML: **Regressão, Classificação,** Clusterização, Regras de Associação, Detecção de Anomalias, Matching etc.
- CRISP DM, 6 fases, **não linear** -> As fases se conversam, vou e volto entre elas 
- Aprendizado: -> O que é cada tipo
  - **Supervisionado:** **Regressão, Classificação** (Conjunto de Treinamento: Exemplos)
  - Não Supervisionado
  - Com Reforço 

**2** [**Python básico para o Aprendizado de Máquina**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Python_1.ipynb) 

- Listas
- Dicionários

**3** [**Python Pandas e Matplotlib**](https://colab.research.google.com/github/Rogerio-mack/Inteligencia_Artificial/blob/main/IA_Python_2.ipynb)

- `Pandas` 
  - Seleção de Dados: `df[ <predicado lógico> ][ <lista de colunas ]`, `tips[ df.tip > df.tip.mean() ][['total_bill','tip','sex']]`
  - `nlargest()` `nsmallest`
  - `pd.merge()`
  - `df.groupby()`
  - `df.describe(include ='all')`

**4** [**Aprendizado Supervisionado e Regressão Linear**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML2_Regressao.ipynb)

- Aprendizado Supervisionado $\rightarrow$ Conjunto de Treinamento, Dados rotulados
- **Esquema Geral dos Modelos Supervisionados**
- Regressão $\times$ |Classificação
- Regressão Linear Simples e Múltipla $y = a_0 + a_1 x_1 + ... + a_n x_n$ **Estes coeficientes diminuem a reta**
- Regressão Linear: Transformações da variável **dependente**, **posso**, já das **preditoras (ou independentes), Não posso**
- Coeficientes $\rightarrow$ Minimização do Erro
- Coeficiente de Determinação, **$R2$** -> 
  - R2 ajustado: None! -> Não cai
  - R2 $\in [0,1]$
  - p-value $< 0.05$ para os coeficientes -> Se o coeficiente é ou não significativo, precisar ser < 0.05 para ser significativo
  - R2 = 0? Não há relação? -> Não há relação linear, mas pode haver relação 
  - R2 = 1? Causa-Efeito? -> Não, 
 - Variáveis Categóricas $\rightarrow$ Hot Encode nelas para poder usar 
 - Simple Code: `model = sm.ols('y ~ x',data=df); r = model.fit()` -> só preciso entender isso (independente aparece com o +

**5** [**Classificação: Regressão Logística**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML3_RegressaoLogistica.ipynb)

- **Esquema Geral dos Modelos Supervisionados**
- Tenho uma regressão linear onde eu apliquei uma logística
- Estimando os parâmetros, quais?
- **Esquema Geral dos Estimadores no `scikit-learn`

```
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

X = df[['x1','x2']]
y = df.y

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=123)

clf = LogisticRegression(max_iter=1000)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print( clf.score(X_test,y_test) )
```

- Regressão Logística: **Ele é Classificador ou Separador Linear**
- Regressão Logística: Normalização $\leftarrow$ sim, é sensível à normalização
- Regressão Logística: **Classificador binário**
- Regressão Logística: variáveis preditoras numéricas, mas a dependente não necessariamente
- Dilema Viés-Variância: **Underfitting $\times$ Overfitting**
- Evitando Underfitting? 
- Evitando Overfitting? **Conjuntos de Treinamento e Teste**
- Acuracidade e Risco dessa métrica (classes desbalanceadas)
- *Um modelo de Deep Learning é sempre melhor que um modelo simples de Regressão Logística que só pode classificar corretamente dados linearmente separáveis.* True or False?

**6** [**Métricas de Classificação e K-Vizinhos mais Próximos**](https://colab.research.google.com/github/Rogerio-mack/Machine-Learning-I/blob/main/ML4_Knn.ipynb)

- Esquema geral do Knn
- **Matriz de Confusão, Precisão e Recall**

**7** **Proposta de Projeto, na data da prova**.

- Definição do Problema, Recursos e Referência (=uma atividade peso 2).

