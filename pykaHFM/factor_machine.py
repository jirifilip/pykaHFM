import numpy as np

from .data_transform import DataTransformer

class FactorizationMachine:
    """
    A Factorization Machine class 
    
    """
    
    def __init__(self, users, items, y, v_matrix):
        self.users = users
        self.items = items
        self.n_users = len(users)
        self.n_items = len(items)
        self.y = y
        
        self.data_transformer = DataTransformer(users, items, y)
        self.training_data, self.training_y = self.data_transformer.transform()
        
        self.w_bias = np.random.normal()
        self.w_pars = np.random.normal(size=self.n_users + self.n_items)
        self.v_matrix = np.copy(v_matrix)
        
        
    
    def predict(self, user_idx, item_idx):
        n = self.training_data.shape[0]
        n_users = len(self.users)
        n_items = len(self.items)
        
        x_hat = self.w_bias + self.w_pars[user_idx] + self.w_pars[n_users + item_idx] + self.v_matrix[user_idx, :]@self.v_matrix[user_idx + item_idx, :].T
        
        return x_hat

    def predict_all(self):
        n = self.training_data.shape[0]
        n_users = len(self.users)
        n_items = len(self.items)

        X_hat = np.zeros_like(self.y)
        for user_idx in range(n_users):
            for item_idx in range(n_items):
                x_hat = self.predict(user_idx, item_idx)
    
                X_hat[user_idx, item_idx] = x_hat

        return X_hat
    
    def loss_MSE(self):
        
        y_hats = []
        
        for user_idx in range(len(self.users)):
            for item_idx in range(len(self.items)):
                y_hat = self.predict(user_idx, item_idx)
                y_hats.append(y_hat)
               
            
        y_hats = np.array(y_hats)
               
        mse = np.mean((y_hats - self.training_y)**2)
            
        return 1/2 * mse
    
    
    def loss_MSE_gradient(self, x_input):
        n_features = self.v_matrix.shape[1]
        n = self.v_matrix.shape[0]
        
        w_bias_gradient = -1
        w_pars_gradient = -x_input
        
        v_gradients = np.zeros_like(self.v_matrix)

        for idx_i in range(n):
            x_i = x_input[idx_i]
            x_i_vector = np.array([x_i**2] * n)
            
            for feature_idx in range(n_features):
                
                grad_sum = np.dot(self.v_matrix[:, feature_idx], x_input) - np.dot(self.v_matrix[:, feature_idx], x_i_vector)
                
                v_grad = x_i * grad_sum
                v_gradients[idx_i, feature_idx] = -v_grad
        
        
        
        
        return w_bias_gradient, w_pars_gradient, v_gradients