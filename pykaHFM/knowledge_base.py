def generate_knowledge_base(items, predicates):
    knowledge_base = []
    
    for item in items:
        for predicate in predicates.keys():
            options = predicates[predicate]
            n_options = len(options)
            
            object_subset = np.random.choice(options, np.random.randint(1, n_options + 1))
            for chosen_object in object_subset:
                knowledge_base.append((item, predicate, chosen_object))
    return knowledge_base