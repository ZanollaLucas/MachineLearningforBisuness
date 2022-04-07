
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt


films = pd.read_csv("movies.csv")
films.columns = ["filmId", "title", "gender"]
films = films.set_index("filmId")
print(films.head())

ratings = pd.read_csv("ratings.csv")
ratings.columns = ["userId", "filmId", "rating", "timestamp"]
ratings.head()

print(ratings.describe())

"""# Primeira tentativa de recomendação: heurística de total de votos"""

rating_counts = ratings["filmId"].value_counts()
rating_counts.head()

films['rating_counts'] = rating_counts
films.head()

films.sort_values("rating_counts", ascending = False).head(10)

ratings_medias = ratings.groupby("filmId").mean()["rating"]
ratings_medias.head()

films["rating_media"] = ratings_medias
films.sort_values("rating_counts", ascending = False).head(10)

films.sort_values("rating_media", ascending = False).head(10)

films.query("rating_counts >= 10").sort_values("rating_media", ascending = False).head(10)

films_with_50_more_counts = films.query("rating_counts >= 50")
films_with_50_more_counts.sort_values("rating_media", ascending = False).head(10)

ich_habe_geschautet = [1, 21, 19, 10, 11, 7, 2]
films.loc[ich_habe_geschautet]

aventura_infantil_e_fantasia = films_with_50_more_counts.query("gender=='Adventure|Children|Fantasy'")
aventura_infantil_e_fantasia.drop(ich_habe_geschautet, errors='ignore').sort_values("rating_media", ascending = False).head(10)

def pitagoras(a,b):
  (delta_x, delta_y) = a - b
  return sqrt(delta_x * delta_x + delta_y * delta_y)



def pitagoras(a,b):
  return np.linalg.norm(a - b)


"""# Distance between users"""

def vector_distance(a,b):
  return np.linalg.norm(a - b)

def user_ratings(user):
  user_ratings = ratings.query("userId==%d" % user)
  user_ratings = user_ratings[["filmId", "rating"]].set_index("filmId")
  return user_ratings

user1 = user_ratings(1)
user4 = user_ratings(4)

user1.head()

user4.head()

difference = user1.join(user4, lsuffix="_left", rsuffix="_right").dropna()
vector_distance(difference['rating_left'], difference['rating_right'])

def users_distance(user_id1, user_id2):
  ratings1 = user_ratings(user_id1)
  ratings2 = user_ratings(user_id2)
  difference = ratings1.join(ratings2, lsuffix="_left", rsuffix="_right").dropna()
  distance =  vector_distance(difference['rating_left'], difference['rating_right'])
  return [user_id1, user_id2, distance]

users_distance(1,4)

numeber_users = len(ratings['userId'].unique())
print("Temos %d users" % numeber_users)

def distance_alles(your_id):
  distancies = []
  for user_id in ratings['userId'].unique():
    info = users_distance(your_id, user_id)
    distancies.append(info)
  return distancies

distance_alles(1)[:5]

def distance_alles(your_id):
  all_users = ratings['userId'].unique()
  distancies = [users_distance(your_id, user_id) for user_id in all_users]
  distancies = pd.DataFrame(distancies, columns = ["your", "another_p", "distance"])
  return distancies

distance_alles(1).head()

user_ratings(1).join(user_ratings(5), lsuffix="_1", rsuffix="5").dropna()

user_ratings(1).join(user_ratings(2), lsuffix="_1", rsuffix="2").dropna()

user_ratings(1).join(user_ratings(3), lsuffix="_1", rsuffix="3").dropna()

"""# Usuários sem films em comum são colocados bem distante um do outro"""

def users_distance(user_id1, user_id2, minimo = 5):
  ratings1 = user_ratings(user_id1)
  ratings2 = user_ratings(user_id2)
  difference = ratings1.join(ratings2, lsuffix="_left", rsuffix="_right").dropna()
  
  if(len(difference) < minimo):
    return [user_id1, user_id2, 100000]
  
  distance =  vector_distance(difference['rating_left'], difference['rating_right'])
  return [user_id1, user_id2, distance]

distance_alles(1).head()

def closer_to(your_id):
  distancies = distance_alles(your_id)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id)
  return distancies

closer_to(1).head()

"""# Parâmetros para teste"""

def closer_to(your_id, n = None):
  distancies = distance_alles(your_id, n = n)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id)
  return distancies

def distance_alles(your_id, n = None):
  all_users = ratings['userId'].unique()
  if n:
    all_users = all_users[:n]
  distancies = [users_distance(your_id, user_id) for user_id in all_users]
  distancies = pd.DataFrame(distancies, columns = ["your", "another_p", "distance"])
  return distancies

closer_to(1, n = 50)

def users_distance(user_id1, user_id2, minimo = 5):
  ratings1 = user_ratings(user_id1)
  ratings2 = user_ratings(user_id2)
  difference = ratings1.join(ratings2, lsuffix="_left", rsuffix="_right").dropna()
  
  if(len(difference) < minimo):
    return None
  
  distance =  vector_distance(difference['rating_left'], difference['rating_right'])
  return [user_id1, user_id2, distance]

def distance_alles(your_id, user_to_analyse = None):
  all_users = ratings['userId'].unique()
  if user_to_analyse:
    all_users = all_users[:user_to_analyse]
  distancies = [users_distance(your_id, user_id) for user_id in all_users]
  distancies = list(filter(None, distancies))
  distancies = pd.DataFrame(distancies, columns = ["your", "another_p", "distance"])
  return distancies

def closer_to(your_id, user_to_analyse = None):
  distancies = distance_alles(your_id, user_to_analyse = user_to_analyse)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id)
  return distancies

closer_to(1, user_to_analyse = 50)

def recommend(your, user_to_analyse = None):
  your_ratings = user_ratings(your)
  your_films = your_ratings.index

  similars = closer_to(your, user_to_analyse = user_to_analyse)
  similar = similars.iloc[0].name
  rsimilar_ratings = user_ratings(similar)
  rsimilar_ratings = rsimilar_ratings.drop(your_films, errors='ignore')
  recommendations = rsimilar_ratings.sort_values("rating", ascending=False)
  return recommendations.join(films)

recommend(1, user_to_analyse=50).head()

recommend(1).head()

"""# Sugerindo baseado em vários usuários"""

def closer_to(your_id, n_closers=10, user_to_analyse = None):
  distancies = distance_alles(your_id, user_to_analyse = user_to_analyse)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id)
  return distancies.head(n_closers)

closer_to(1, n_closers = 2, user_to_analyse=300)

def recommend(your, n_closers = 10, user_to_analyse = None):
  your_ratings = user_ratings(your)
  your_films = your_ratings.index

  similars = closer_to(your, n_closers = n_closers, user_to_analyse = user_to_analyse)
  users_similars = similars.index
  similar_ratings = ratings.set_index("userId").loc[users_similars]
  recommendations = similar_ratings.groupby("filmId").mean()[["rating"]]
  recommendations = recommendations.sort_values("rating", ascending=False)
  return recommendations.join(films)

recommend(1, user_to_analyse = 50).head()

recommend(1, user_to_analyse = 300).head()

recommend(1).head()

def knn(your_id, k_closers=10, user_to_analyse = None):
  distancies = distance_alles(your_id, user_to_analyse = user_to_analyse)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id)
  return distancies.head(k_closers)

def recommend(your, k_closers = 10, user_to_analyse = None):
  your_ratings = user_ratings(your)
  your_films = your_ratings.index

  similars = knn(your, k_closers = k_closers, user_to_analyse = user_to_analyse)
  users_similars = similars.index
  similar_ratings = ratings.set_index("userId").loc[users_similars]
  recommendations = similar_ratings.groupby("filmId").mean()[["rating"]]
  recommendations = recommendations.sort_values("rating", ascending=False)
  return recommendations.join(films)

"""# Test"""

films.loc[[122904, 1246, 2529, 2329 , 2324 , 1 , 7 , 2 ,1196, 260]]

def new_user(dados):
  new_user = ratings['userId'].max()+1
  user_ratings_new = pd.DataFrame(dados, columns=["filmId", "rating"])
  user_ratings_new['userId'] = new_user
  return pd.concat([ratings, user_ratings_new])

ratings = new_user([[122904,2],[1246,5],[2529,2],[2329,5],[2324,5],[1,2],[7,0.5],[2,2],[1196,1],[260,1]])
ratings.tail()

recommend(8621).head()

"""# Utilizar somente as ratings de films com mais de 50 votos"""

ratings = ratings.set_index("filmId").loc[films_with_50_more_counts.index]
ratings.head()

ratings = ratings.reset_index()
ratings.head()

recommend(8621).head()

def recommend(your, k_closers = 10, user_to_analyse = None):
  your_ratings = user_ratings(your)
  your_films = your_ratings.index

  similars = knn(your, k_closers = k_closers, user_to_analyse = user_to_analyse)
  users_similars = similars.index
  similar_ratings = ratings.set_index("userId").loc[users_similars]
  recommendations = similar_ratings.groupby("filmId").mean()[["rating"]]
  countings = similar_ratings.groupby("filmId").count()[['rating']]
  
  minimum = k_closers / 2
  recommendations = recommendations.join(countings, lsuffix="_media_dos_users", rsuffix="_countings_nos_users")
  recommendations = recommendations.query("rating_countings_nos_users >= %.2f" % minimum)  
  recommendations = recommendations.sort_values("rating_media_dos_users", ascending=False)
  recommendations = recommendations.drop(your_films,errors='ignore')
  return recommendations.join(films)

def knn(your_id, k_closers=10, user_to_analyse = None):
  distancies = distance_alles(your_id, user_to_analyse = user_to_analyse)
  distancies = distancies.sort_values("distance")
  distancies = distancies.set_index("another_p").drop(your_id, errors='ignore')
  return distancies.head(k_closers)

recommend(1, user_to_analyse=500)

recommend(8621).head(10)

recommend(8621, k_closers=20).head(10)