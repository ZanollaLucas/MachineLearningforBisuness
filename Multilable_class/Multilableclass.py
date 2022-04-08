import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.adapt import MLkNN


perguntas = pd.read_csv("stackoverflow_perguntas.csv")
perguntas.sample(10)
print(len(perguntas))
perguntas.Tags.unique()
lista_de_tags = list()
for tags in perguntas.Tags.unique():
    for tag in tags.split():
        if tag not in lista_de_tags:
            lista_de_tags.append(tag)
        
print(lista_de_tags)

node_js = list()
for linha_tag in perguntas.Tags:
    if "node.js" in linha_tag:
        node_js.append(1)
    else:
        node_js.append(0)
perguntas["node.js"] = node_js
perguntas

def nova_coluna(lista_tags, dataframe, nome_tags):
    for tag in lista_tags:
        coluna = list()
        for linha_tag in dataframe[nome_tags]:
            if tag in linha_tag:
                coluna.append(1)
            else:
                coluna.append(0)
        dataframe[tag] = coluna
nova_coluna(lista_de_tags, perguntas, "Tags")
perguntas.sample(10)

perguntas_treino, perguntas_test, tags_treino, tags_teste = train_test_split(
    perguntas.Peguntas,
    perguntas.Tags
)

lista_1 = [1,2]
lista_2 = [5,4]
lista_zip = zip(lista_1, lista_2)
print(list(lista_zip))

lista_zip_tags = list(zip(perguntas[lista_de_tags[0]],
                     perguntas[lista_de_tags[1]],
                     perguntas[lista_de_tags[2]],
                     perguntas[lista_de_tags[3]]))

perguntas["todas_tags"] = lista_zip_tags
perguntas.sample(10)

perguntas_treino, perguntas_test, tags_treino, tags_teste = train_test_split(
    perguntas.Perguntas,
    perguntas.todas_tags,
    test_size = 0.2,
    random_state = 123
)

print(perguntas_treino)

vetorizar = TfidfVectorizer(max_features=5000, max_df=0.85)
print(vetorizar)

vetorizar.fit(perguntas.Perguntas)
perguntas_treino_tfidf = vetorizar.transform(perguntas_treino)
perguntas_test_tfidf = vetorizar.transform(perguntas_test)
print(perguntas_treino_tfidf.shape)
print(perguntas_test_tfidf.shape)

regressao_logistica = LogisticRegression()
classificador_onevsrest = OneVsRestClassifier(regressao_logistica)
classificador_onevsrest.fit(perguntas_treino_tfidf, tags_treino)

tags_treino_array = np.asarray(list(tags_treino))
tags_test_array = np.asarray(list(tags_teste))
tags_treino_array
type(tags_treino)

regressao_logistica = LogisticRegression()
classificador_onevsrest = OneVsRestClassifier(regressao_logistica)
classificador_onevsrest.fit(perguntas_treino_tfidf, tags_treino)
type(tags_treino)

tags_treino_array = np.asarray(list(tags_treino))
tags_teste_array = np.asarray(list(tags_teste))
print(tags_treino_array)
print(type(tags_treino_array))

regressao_logistica = LogisticRegression(solver = 'lbfgs')
classificador_onevsrest = OneVsRestClassifier(regressao_logistica)
classificador_onevsrest.fit(perguntas_treino_tfidf, tags_treino_array)
resultado_onevsrest = classificador_onevsrest.score(perguntas_test_tfidf, tags_teste_array)
print("Resultado {0: .2f}%".format(resultado_onevsrest*100))

perguntas.todas_tags.unique()

len(perguntas.todas_tags.unique())

previsao_onevsrest = classificador_onevsrest.predict(perguntas_test_tfidf)
hamming_loss_onevsrest = hamming_loss(tags_teste_array, previsao_onevsrest)
print("Hamming Loss {0: .2f}".format(hamming_loss_onevsrest))

perguntas.corr()

classificador_cadeia = ClassifierChain(regressao_logistica)
classificador_cadeia.fit(perguntas_treino_tfidf, tags_treino_array)
resultado_cadeia = classificador_cadeia.score(perguntas_test_tfidf, tags_teste_array)
previsao_cadeia = classificador_cadeia.predict(perguntas_test_tfidf)
hamming_loss_cadeia = hamming_loss(tags_teste_array, previsao_cadeia)
print("Hamming Loss {0: .2f}".format(hamming_loss_cadeia))
print("Resultado {0: .2f}%".format(resultado_cadeia*100))

classificador_br = BinaryRelevance(regressao_logistica)
classificador_br.fit(perguntas_treino_tfidf, tags_treino_array)
resultado_br = classificador_br.score(perguntas_test_tfidf, tags_teste_array)
previsao_br = classificador_br.predict(perguntas_test_tfidf)
hamming_loss_br = hamming_loss(tags_teste_array, previsao_br)
print("Hamming Loss {0: .2f}".format(hamming_loss_br))
print("Resultado {0: .2f}%".format(resultado_br*100))

classificador_mlknn = MLkNN()
classificador_mlknn.fit(perguntas_treino_tfidf, tags_treino_array)
resultado_mlknn = classificador_mlknn.score(perguntas_test_tfidf, tags_teste_array)
previsao_mlknn = classificador_mlknn.predict(perguntas_test_tfidf)
hamming_loss_mlknn = hamming_loss(tags_teste_array, previsao_cadeia)
print("Hamming Loss {0: .2f}".format(hamming_loss_mlknn))
print("Resultado {0: .2f}%".format(resultado_mlknn*100))

print("Hamming Loss cadeia {0: .2f}".format(hamming_loss_cadeia))
print("Resultado cadeia {0: .2f}%".format(resultado_cadeia*100))

print("Hamming Loss br {0: .2f}".format(hamming_loss_br))
print("Resultado br {0: .2f}%".format(resultado_br*100))

resultados_classificacao = pd.DataFrame()
resultados_classificacao["perguntas"] = perguntas_test.values
resultados_classificacao["tags real"] = list(tags_teste)
resultados_classificacao["BR"] = list(previsao_br.toarray())
resultados_classificacao["cadeia"] = list(previsao_cadeia.toarray())
resultados_classificacao["mlknn"] = list(previsao_mlknn.toarray())
resultados_classificacao

previsao_br

resultados_classificacao.iloc[1]