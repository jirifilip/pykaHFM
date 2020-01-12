import numpy as np

class TFIDFTransformer:
    
    def __init__(self, knowledge_base_triples, x_matrix, user_names, item_names):
        
        self.x_matrix = x_matrix
        
        self.user_names = user_names
        self.item_names = item_names
        self.n_users, self.n_items = x_matrix.shape
        self.knowledge_base_triples = knowledge_base_triples
        self.all_possible_features_F = list({ (predicate, obj) for
                                             (item, predicate, obj)
                                             in knowledge_base_triples })
        
        self.v_matrix = None
        self.v_matrix_users = None
        self.v_matrix_items = None
        
        
    def generate_v_for_item(self, current_item, feature):
        feature_predicate, feature_object = feature
        tf_numerator = 1 if (current_item, feature_predicate, feature_object) in self.knowledge_base_triples else 0

        tf_denominator = 0
        for current_predicate, current_obj in self.all_possible_features_F:
            predicates = { (predicate, obj) for (item, predicate, obj) in self.knowledge_base_triples
                         if current_item == item and current_predicate == predicate and current_obj == obj }
            n_predicates = len(predicates)

            tf_denominator += n_predicates**2

        tf_denominator = np.sqrt(tf_denominator)

        idf_numerator = self.n_items
        idf_denominator = len({ item for (item, predicate, obj) in self.knowledge_base_triples
                              if feature_predicate == predicate and feature_object == obj })

        v_value = tf_numerator / tf_denominator * np.log(idf_numerator / idf_denominator)

        return v_value
    
    
    def generate_v_for_user(self, user, feature):
        feature_idx = self.all_possible_features_F.index(feature)
        user_idx = self.user_names.index(user)

        items_enjoyed_by_user = [ item_idx 
                                 for item_idx, user_enjoyment 
                                 in enumerate(self.x_matrix[user_idx, :])
                                 if user_enjoyment == 1 ]    

        u_numerator = np.sum(self.v_matrix_items[items_enjoyed_by_user, feature_idx])
        u_denominator = len({ movie_idx for movie_idx in items_enjoyed_by_user
                            if self.v_matrix_items[movie_idx, feature_idx] != 0 })

        if u_denominator == 0:
            return u_denominator

        u_value = u_numerator / u_denominator

        return u_value
    
    
    def generate_v_item_matrix(self):
        v_matrix = np.zeros((self.n_items, len(self.all_possible_features_F)))

        for i, item in enumerate(self.item_names):
            for j, (predicate, obj) in enumerate(self.all_possible_features_F):
                v_matrix[i, j] = self.generate_v_for_item(item, (predicate, obj))

        self.v_matrix_items = v_matrix
        
        
    def generate_v_user_matrix(self):
        u_matrix = np.zeros((self.n_users, len(self.all_possible_features_F)))

        for i, user in enumerate(self.user_names):
            for j, (predicate, obj) in enumerate(self.all_possible_features_F):
                u_matrix[i, j] = self.generate_v_for_user(user, (predicate, obj))

        self.v_matrix_users = u_matrix
        
    def generate_v_matrix(self):
        self.generate_v_item_matrix()
        self.generate_v_user_matrix()
        
        self.v_matrix = np.concatenate([self.v_matrix_users, self.v_matrix_items])