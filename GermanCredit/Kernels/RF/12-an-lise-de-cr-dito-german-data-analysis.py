#!/usr/bin/env python
# coding: utf-8

# # Predição de Análise de Crédito - Marlysson Silva

# ## Tabela de conteúdo
# 
# - Conhecendo o dataset
# - Exploração do Dataset
#     - Tratamento dos dados
#     - Análise descritiva dos dados
#     - Visualizando agrupamentos
#     - Correlação entre variáveis
# - Predição
#     - Algoritmos
#         - Naive Bayes
#         - AdaBoost
#         - OneVsOne
#         - OneVsRest
#         - K-Folding
#         - Random Forest
#         - Logistic Regression

# ## Conhecendo o dataset

# Essa análise se baseará no dataset fornecido pelo repositório de dados UCI Machile Learning Repository, onde pode ser encontrado direto através deste [link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

# ### Informações gerais

# **NOME:** Statlog (German Credit Data) Data Set
# 
# **DESCRIÇÃO: ** Neste dataset da UCI Repository há o conjunto de dados a respeito da qualificação dos clientes de um banco alemão identificando-os como bons ou maus pagadores de empréstimos, ou seja, se são clientes confiáveis para que seja concedido a eles um montante de crédito, com a confiança de que não haverá calote por parte do cliente.
# 
# **DIMENSIONALIDADE DO DATASET:** 1000 Linhas x 21 Colunas

# ### Informações exploratórias

# > **COLUNAS DO DATASET E SEUS SIGNIFICADOS**

# 1. **STATUS OF EXISTING CHECKING ACCOUNT :** Montante existente atualmente na conta.
# 1. **DURATION IN MONTH :** Significa a duração em meses do empréstimo concedido.
# 1. **CREDIT HISTORY :** Informações descritivas sobre o histórico financeiro do cliente. Se os créditos antigos dele estão quitados, se ainda está devendo, se até agora os créditos dele estão em bom estado.
# 1. **PURPOSE :** Propósito destinado para o crédito concedido.
# 1. **CREDIT AMOUNT :** Montante de crédito requisitado ao banco.
# 1. **SAVINGS ACCOUNT/BOUNDS :** Montante guardado na conta poupança.
# 1. **PRESENT EMPLOYMENT SINCE :** Tempo de empregado no atual emprego.
# 1. **INSTALLMENT RATE IN PERCENTAGE OF DISPOSABLE INCOME :** Taxa de de prestação sobre o montante( rendimento ) que o cliente possui.
# 1. **PERSONAL STATUS AND SEX :** Estado civil e sexo do cliente.
# 1. **OTHERS DEBTORS/ GUARANTORS :** Tipo de associação em créditos concedidos que já participou.
# 1. **PRESENT RESIDENCE SINCE :** Tempo de moradia na residência atual.
# 1. **PROPERTY :** Propriedades que possui.
# 1. **AGE IN YEARS :** Idade
# 1. **OTHERS INSTALLMENT PLANS :** Outros empreendimentos que requerem pagamento de prestações.
# 1. **HOUSING :** Tipo de propriedade da residência.
# 1. **NUMBER OF EXISTING CREDITS AT THIS BANK :** Número de créditos já concedidos no banco.
# 1. **JOB :** Estado do trabalho atual
# 1. **NUMBER OF PEOPLE BEING LIABLE TO PROVIDE MAINTENANCE FOR :** Total de dependentes.
# 1. **TELEPHONE :** Indicativo se o cliente possui telefone ou não.
# 1. **FOREIGN WORKER :** Indicando se o cliente é de outra cidade ou se trabalha na mesma cidade do trabalho.
# 1. **STATE RISK :** Coluna indicando se o cliente em questão é um bom cliente para permitir créditos ou não.

# > **OBJETIVO DO DATASET E DA EXPLORAÇÃO DOS DADOS**

# Esse dataset possui alguns atributos com relação ao histórico de crédito do cliente, montante acumulado e algumas outras propriedades com relação à valores monetários.
# 
# Com todos esses atributos, balanceando o nível de confiança do cliente é possível com a aplicação de algoritmos de aprendizagem de máquina descobrir se **há risco ou não com relação ao banco ter prejuízo ao conceder crédito à determinado tipo de cliente**.

# # Exploração do Dataset

# In[ ]:


import pandas as pd


# **OBS:** O dataset provido inicialmente há um problema de "legibilidade" , pois suas colunas não são nomeadas e seus valores estão preenchidos com valores que não dizem muita coisa sobre o que eles significam. Sabendo disso faremos:
# 
# 1. Um tratamento inicialmente nas colunas para identificá-las quanto ao que elas significam
# 1. Mapeamento das siglas usadas para seus reais valores.

# > **RENOMEANDO COLUNAS PARA UAM MELHOR VISUALIZAÇÃO**

# In[ ]:


atributos = ["montante", "duracao", "historico_credito", 
              "proposito", "montante_credito", "poupanca",
              "tempo_empregado","taxa_parcelamento",
              "estado_civil_sexo","tipo_participacao_credito", 
              "tempo_moradia", "propriedade","idade",
              "gastos_adicionais", "habitacao","quantidade_creditos","emprego",
              "dependentes","telefone","trabalhador_estrangeiro","risco"]


# In[ ]:


df = pd.read_csv("../input/credit_approval.txt",header=None, sep=" ",names=atributos)


# In[ ]:


df.head(3)


# ## Transformação dos dados - Limpando dados para melhor visualização

# > **Como o DATAFRAME não possuia os valores reais em suas células é necessário um processamento para visualizar as informações relacionadas ao DATAFRAME, como substituir legendas por valores reais, é isso que iremos fazer agora.**
# 
# 1. Mapear valores das células para melhor descrição
# 2. Renomear todo o dataframe com essas associações

# In[ ]:


codigos_historico_de_creditos = {
    "A30": "no credits taken/all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account/other credits existing (not at this bank)"
}

codigos_proposito = {
    "A40": "car(new)",
    "A41": "car(used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "others"
}

codigo_estado_civil_sexo = {
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/windowed",
    "A95": "female : single"
}

codigos_outros_devedores = {
    "A101": None,
    "A102": "co-applicant",
    "A103": "guarantor",
}

codigos_propriedade = {
    "A121": "real state",
    "A122": "building society/life insurance",
    "A123": "car",
    "A124": "unknown/no property"
}

codigos_planos_de_parcelamento = {
    "A141": "bank",
    "A142": "stores",
    "A143": "None"
}

codigos_residencia = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free"
}

codigos_estado_emprego = {
    "A171": "unemployed/unskilled-non-resident",
    "A172": "unskilled-resident",
    "A173": "skilled employee/official",
    "A174": "management/self-employed/highly qualified employee/officer"
}

codigos_telefone = {
    "A191": None,
    "A192": "yes"
}

codigos_trabalhador_estrangeiro = {
    "A201": "yes",
    "A202": "no"
}


# > **Mapeando variáveis contínuas**

# In[ ]:


codigos_status_atual_conta_corrente = {
    "A11": "< 0",
    "A12": "< 199",
    "A13": ">= 200",
    "A14": None
}

codigos_reserva_poupanca = {
    "A61": "< 100",
    "A62": "< 499",
    "A63": "< 999",
    "A64": ">= 1000",
    "A65": "unknown"
}

codigos_tempo_emprego = {
    "A71": None,
    "A72": "< 1", # Menos de 1 ano
    "A73": "< 4", # Entre 1 ano e menos que 4 anos
    "A74": "< 7", # Entre 4 anos e menos que 7 anos
    "A75": ">= 7" # Mais de 7 anos
}


# In[ ]:


colunas_para_codigos = {
    "montante"             : codigos_status_atual_conta_corrente,
    "historico_credito"    : codigos_historico_de_creditos,
    "proposito"            : codigos_proposito,
    "poupanca"             : codigos_reserva_poupanca,
    "tempo_empregado"      : codigos_tempo_emprego,
    "estado_civil_sexo"    : codigo_estado_civil_sexo, 
    "tipo_participacao_credito"     : codigos_outros_devedores,
    "propriedade"          : codigos_propriedade,
    "gastos_adicionais": codigos_planos_de_parcelamento,
    "habitacao"            : codigos_residencia,
    "emprego"              : codigos_estado_emprego,
    "telefone"             : codigos_telefone,
    "trabalhador_estrangeiro"  : codigos_trabalhador_estrangeiro
}


# > **Mapeando cada colunas associando para seus respectivos valores**

# In[ ]:


df.replace(colunas_para_codigos,inplace=True)


# In[ ]:


df.head(3)


# In[ ]:


df.replace({"unknown":None},inplace=True)


# > **TIPOS DAS COLUNAS DO DATASET**

# In[ ]:


df.dtypes


# > **EXPLORAÇÃO DOS CAMPOS NULOS**

# In[ ]:


df.info()


# ## Tratamento das colunas

# > **CRIAÇÃO DOS CAMPOS: **
# - **estado_civil**
# - **sexo**

# > **JUSTIFICATIVA**
# - Poder fazer correlação se o sexo e o estado civil da pessoa interfere no sucesso ou não da disponibilidade de crédito.
# - Gráficos descritivos para visualizar distribuição dessas categorias perante o sucesso ou não.

# In[ ]:


def criar_sexo_e_estado_civil(coluna):
    dados_separados = coluna.split(":")
    
    sexo = dados_separados[0].strip()
    estado_civil = dados_separados[1].strip()

    return pd.Series([sexo,estado_civil])


# In[ ]:


df[["sexo","estado_civil"]] = df["estado_civil_sexo"].apply(criar_sexo_e_estado_civil)


# > **REMOÇÃO DE COLUNAS: **
#  - **telefone**
#  - **trabalhador_estrangeiro**
#  - **estado_civil_sexo ( coluna decomposta )**
#  - **tipo_participacao_credito**

# > **JUSTIFICATIVA**
# - **Telefone:** não possui impacto significativo em estar ou não no dataset, visto que são 2 possíveis valores
# - **Trabalhador estrangeiro:** não possui muitos valores que possam impactar no resultado final
# - **Estado civil e sexo:** foi removida pois foi decomposta em outras colunas no dataset
# - **Tipo participacao credito**: foi removida pois não dá mais detalhes sobre o dado... somente uma descrição e nada mais. Juguei desnecessário o uso.

# In[ ]:


colunas = ["telefone","trabalhador_estrangeiro", "estado_civil_sexo","gastos_adicionais","tipo_participacao_credito"]
df = df.drop(colunas,axis=1)


# In[ ]:


df.dtypes


# ## Análise descritiva dos dados

# In[ ]:





# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# > **AGRUPAMENTO DA DISTRIBUIÇÃO DAS QUANTIDADES DE MONTANTES**

# In[ ]:


a = sns.countplot(x="montante",data=df)
a.set_title("Contagem do montante por tipo")


# > **ANÁLISE DE CORRELAÇÃO ENTRE AS VARIÁVEIS DO DATASET**

# In[ ]:


plt.figure(figsize=(15, 5))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.swarmplot(x="historico_credito",y="montante_credito",data=df)


# In[ ]:


plt.figure(figsize=(15, 5))
sns.distplot(df.idade)


# In[ ]:


df.head()


# # Limpando os dados para aplicação dos algoritmos

# In[ ]:


def mapear_valores(coluna):
    valores = tuple(set(df[coluna].values))

    associados = tuple(range(len(valores)))

    df[coluna].replace(valores,associados,inplace=True)


# In[ ]:


colunas = ["historico_credito","montante","proposito","poupanca",
           "tempo_empregado","propriedade","habitacao",
           "emprego","sexo","estado_civil"]

for coluna in colunas:
    mapear_valores(coluna)


# # Aplicando algoritmos

# > **SEPARAÇÃO DOS DADOS PARA TREINO E TESTE**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x = df.drop('risco', 1).values
y = df["risco"].values

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)


# ### Algoritmo Naive Bayes

# In[ ]:


def aplicar_modelo(modelo, x_treino, y_treino, x_teste, y_teste):
    
    modelo.fit(x_treino,y_treino)
    
    risco = modelo.predict(x_teste)
    
    return accuracy_score(y_teste,risco)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

naive = MultinomialNB()

resultado = aplicar_modelo(naive,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("Naive Bayes: {}".format(resultado))


# ### Algoritmo Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier()

resultado = aplicar_modelo(ada_boost,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("Ada boost: {}".format(resultado))


# ### Algoritmo RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

resultado = aplicar_modelo(random_forest,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("Random Forest: {}".format(resultado))


# ### Algoritmo Regressão Logística

# In[ ]:


from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

resultado = aplicar_modelo(logistic_regression,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("Regressão Logística: {}".format(resultado))


# ### Algoritmo OneVsOne

# In[ ]:


from sklearn.svm import LinearSVC


# In[ ]:


from sklearn.multiclass import OneVsOneClassifier


# In[ ]:


one_vs_one = OneVsOneClassifier(LinearSVC(random_state = 0))

resultado = aplicar_modelo(one_vs_one,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("One vs One classifier: {}".format(resultado))


# ### Algoritmo OneVsRest

# In[ ]:


from sklearn.multiclass import OneVsRestClassifier

one_vs_rest = OneVsRestClassifier(LinearSVC(random_state = 0))

resultado = aplicar_modelo(one_vs_rest,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("One vs Rest classifier : {}".format(resultado))


# ### Aplicando KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

resultado = aplicar_modelo(knn,x_treino,y_treino, x_teste,y_teste)


# In[ ]:


print("KNN classifier: {}".format(resultado))


# ### Aplicando K-folding

# In[ ]:


from sklearn.cross_validation import cross_val_score
import numpy as np

algoritmos = [MultinomialNB(), AdaBoostClassifier(), 
              RandomForestClassifier(), LogisticRegression(), 
              OneVsOneClassifier(LinearSVC(random_state = 0)),
              OneVsRestClassifier(LinearSVC(random_state = 0)),
              KNeighborsClassifier()
             ]

resultados = []

k_folding = len(df.columns) // 2

for modelo in algoritmos:
    
    resultado = cross_val_score(modelo,x_treino,y_treino,cv=k_folding)
    resultados.append(np.mean(resultado))
    
    print("Algoritmo: {}\n Resultado: {:.2f}\n".format(str(modelo.__class__).split(".")[-1], np.mean(resultado)))


# In[ ]:


resultados_series = pd.Series(resultados, index=['Naive Bayes','AdaBoostClassifier',
                                       'RandomForestClassifier','LogisticRegression',
                                       'OneVsOneClassifier','OneVsRestClassifier',
                                      'KNeighborsClassifier'])


# In[ ]:


resultados_series.plot(kind="bar")


# In[ ]:




