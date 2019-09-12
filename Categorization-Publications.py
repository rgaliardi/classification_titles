#!/usr/bin/env python
# coding: utf-8

# ## 1. Importação de Biblotecas de Apoio

# In[1]:


#!pip install --upgrade pip --user


# ## 2. Preparação do Ambiente

# In[2]:


# Carrega as bibliotecas de ambiente

import os
import io
import requests
import collections

path = os.getcwd()

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


# Carrega as bibliotecas de ciências e gráficos

from numba import vectorize

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import warnings
from sklearn import metrics

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 3. Funções de Apoio

# In[4]:


# Remove espaços em branco

@vectorize
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


# ## 4. Coleta de Dados

# In[ ]:


# Coleta dos Dados de treino

# urltrain = "https://meli-data-challenge.s3.amazonaws.com/train.csv.gz"
# ctrain = requests.get(urltrain).content
# ftrain = pd.read_csv(io.StringIO(ctrain.decode('utf-8')), compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False, skipinitialspace=True)
dftrain = pd.read_csv("input/train.csv", header=0, sep=',', quotechar='"', error_bad_lines=False, skipinitialspace=True)
dftrain.head(5)


# In[ ]:


# Coleta dos Dados de teste

# urltest = "https://meli-data-challenge.s3.amazonaws.com/test.csv"
# ctest = requests.get(urltest).content
# ftest = pd.read_csv(io.StringIO(ctest.decode('utf-8')), header=0, sep=',', quotechar='"', error_bad_lines=False, skipinitialspace=True)
dftest = pd.read_csv("input/test.csv", header=0, sep=',', quotechar='"', error_bad_lines=False, skipinitialspace=True)
dftest.head(5)


# ## 4. Processamento/Tratamento de Dados

# ### 4.1 - Processamento de tratamento dos dados de treino

# In[ ]:


# Verifica os dados carregados

dftrain.describe() 


# In[ ]:


# Verificação das caracteristicas de cada coluna do arquivo

dftrain.info()


# In[ ]:


# Verifica se exitem dados nulos no geral

dftrain.isnull().values.any() 


# In[ ]:


# Verifica se exitem dados nulos por coluna

dftrain[dftrain.isnull().any(axis=1)] 


# ### 4.2 - Processamento de tratamento dos dados de teste

# In[ ]:


# Verifica os dados carregados

dftest.describe()


# In[ ]:


# Verificação das caracteristicas de cada coluna do arquivo

dftest.info()


# In[ ]:


# Verifica se exitem dados nulos no geral

dftest.isnull().values.any() 


# In[ ]:


# Verifica se exitem dados nulos por coluna

dftest[dftest.isnull().any(axis=1)] 


# ## 5. Análise e Exploração dos Dados

# ### 5.1 Análise dos Dados

# In[ ]:


# Gráfico com os dados de cada coluna

columns=dftrain.columns[:4]
plt.subplots(figsize=(18,15))
length=len(columns)

for i,j in zip(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    dftrain[i].hist(bins=20,edgecolor='black')
    plt.title(i)
    
plt.show()


# In[ ]:


# Gráfico de boxplot para análise dos dados colunáres

dftrain.plot(kind= 'box', subplots=True, layout=(3,3),figsize=(14,10))


# In[ ]:


# Analise dos dados relacionados a variável target e a sua frequência

dftrain['Category'].value_counts().plot(kind='bar', figsize=(6,6))
plt.title('MercadoLibre - Category')
plt.xlabel('Category')
plt.ylabel('Frequency')

plt.show()


# ### 5.2 - Relacionamento Entre Atributos

# In[ ]:


# Gráfico de boxplot por relacionamento entre Title e Category

dftrain.boxplot(column='title',by='category')

plt.show()


# In[ ]:


# Gráfico de boxplot por relacionamento entre Label Quality e Category

dftrain.boxplot(column='label_quality',by='category')

plt.show()


# In[ ]:


# Gráfico de boxplot por relacionamento entre Language e Category

dftrain.boxplot(column='language',by='category')

plt.show()


# ### 5.3 - Matrix de Correlação

# In[ ]:


# Gráfico com a matrix de correlação entre as variáveis

corr = dftrain.corr()
_ , ax = plt.subplots( figsize =( 12 , 10 ) )
cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })


# In[ ]:


# Gráfico de relacionamento entre as colunas e suas correlações

import seaborn as sns

sns.pairplot(dftrain,hue='category',palette='coolwarm')


# ## 6. Preparração dos dados para aplicação dos Modelos de Machine Learning

# In[ ]:


# Contagem dos valores da variável target - category

df.category.value_counts()


# In[ ]:


# Criando a variável para manter a distribuição sempre padrão

random_state=1143795

