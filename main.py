import numpy as np
import pandas as pd
from collections import Counter

from pykaHFM import TFIDFTransformer, FactorizationMachine, StochasticGradientDescent, load_knowledge_base_triples


knowledge_base_sparql_triples, knowledge_base_new = load_knowledge_base_triples("c:/Users/jfili/Downloads/sparql (4)", ["movie", "subject"])

movies = list(set([subj for subj, *_ in knowledge_base_sparql_triples]))
users = [ "user{}".format(idx) for idx in range(4) ]
n_movies = len(movies)
n_users = len(users)

user_movie_matrix = np.random.randint(0, 2, size=(n_users, n_movies))

tfidf = TFIDFTransformer(knowledge_base_new, user_movie_matrix, users, movies)

tfidf.generate_v_matrix()


fm = FactorizationMachine(users, movies, user_movie_matrix, tfidf.v_matrix)
fm.build_training_data()


sgd = StochasticGradientDescent(fm, iterations=10, learning_rate=0.0001)
loss1 = sgd.fit()

print(loss1)