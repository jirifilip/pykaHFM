# pykaHFM

Implementace algoritmu kaHFM.

```python
import numpy as np

from pykaHFM import (
    TFIDFTransformer,
    FactorizationMachine,
    StochasticGradientDescent,
    load_knowledge_base_triples,
)

movies = ["Avengers", "Avengers: Infinity War",
          "Stardust", "Princess Bride", "Last Witchhunter", "Tall Girl"]
users = ["Chandler", "Joey", "Ross", "Monica", "Rachel", "Phoebe"]


n_movies = len(movies)
n_users = len(users)

user_movie_matrix = np.random.randint(1, 6, size=(n_users, n_movies))
knowledge_base = [('Avengers', 'genre', 'Teen'),
 ('Avengers', 'genre', 'Thriller'),
 ('Avengers', 'genre', 'Comedy'),
 ('Avengers', 'genre', 'Fantasy'),
 ('Avengers', 'genre', 'Comedy'),
 ('Avengers', 'genre', 'Adult'),
 ('Avengers', 'genre', 'Adult'),
 ('Avengers: Infinity War', 'genre', 'Adult'),
 ('Avengers: Infinity War', 'genre', 'Teen'),
 ('Avengers: Infinity War', 'genre', 'Adventure'),
 ('Avengers: Infinity War', 'genre', 'Thriller'),
 ('Stardust', 'genre', 'Comedy'),
 ('Stardust', 'genre', 'Adventure'),
 ('Stardust', 'genre', 'Drama'),
 ('Stardust', 'genre', 'Adult'),
 ('Stardust', 'genre', 'Comedy'),
 ('Stardust', 'genre', 'Drama'),
 ('Princess Bride', 'genre', 'Adventure'),
 ('Last Witchhunter', 'genre', 'Fantasy'),
 ('Last Witchhunter', 'genre', 'Drama'),
 ('Last Witchhunter', 'genre', 'Action'),
 ('Tall Girl', 'genre', 'Teen'),
 ('Tall Girl', 'genre', 'Adult'),
 ('Tall Girl', 'genre', 'Thriller'),
 ('Tall Girl', 'genre', 'Adventure'),
 ('Tall Girl', 'genre', 'Teen'),
 ('Tall Girl', 'genre', 'Teen'),
 ('Tall Girl', 'genre', 'Drama'),
 ('Tall Girl', 'genre', 'Sci-fi')
 ]

tfidf = TFIDFTransformer(knowledge_base, user_movie_matrix, users, movies)
tfidf.generate_v_matrix()

fm = FactorizationMachine(users, movies, user_movie_matrix, tfidf.v_matrix)

sgd = StochasticGradientDescent(fm, iterations=10, learning_rate=0.0001)
sgd.fit()

Y_hat = fm.predict_all()

print(Y_hat)


```