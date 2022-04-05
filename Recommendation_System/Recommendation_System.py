
import pandas as pd


filmes = pd.read_csv("movies.csv")
print(filmes.head())
notas = pd.read_csv("ratings.csv")
print(notas.describe())