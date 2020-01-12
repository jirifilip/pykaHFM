from collections import Counter
import pandas as pd

def load_knowledge_base_triples(csv_path, cols):

    knowledge_base_data = pd.read_csv(csv_path)
    knowledge_base_data["genre"] = "genre"
    knowledge_base_data = knowledge_base_data[["movie", "genre", "subject"]]

    knowledge_base_sparql_triples = knowledge_base_data.values

    a = [ (predicate, obj) for (item, predicate, obj) in knowledge_base_sparql_triples ]
    ctr = Counter(a)
    features_only = [ feature for (feature, _) in ctr.most_common(25) ]
    knowledge_base_new = [ (subj, pred, obj) for (subj, pred, obj) in knowledge_base_sparql_triples if (pred, obj) in features_only ]

    return knowledge_base_sparql_triples, knowledge_base_new