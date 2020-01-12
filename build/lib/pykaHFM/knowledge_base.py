movies = ["Avengers", "Avengers: Infinity War",
          "Stardust", "Princess Bride", "Last Witchhunter", "Tall Girl"]
users = ["Kateřina", "Jirka", "Václav", "Jan"]

predicates = dict(
    genre=["Fantasy", "Comedy", "Action", "Adventure", "Drama", "Sci-fi", "Thriller", "Teen", "Adult"]
)

def generate_knowledge_base(items):
    knowledge_base = []
    
    for item in items:
        for predicate in predicates.keys():
            options = predicates[predicate]
            n_options = len(options)
            
            object_subset = np.random.choice(options, np.random.randint(1, n_options + 1))

            for chosen_object in object_subset:
                knowledge_base.append((item, predicate, chosen_object))

    return knowledge_base

knowledge_base = generate_knowledge_base(movies)