import numpy as np

class StochasticGradientDescent:
    
    def __init__(self, factorization_machine, iterations=500, learning_rate=0.01):
        self.factorization_machine = factorization_machine
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        
    def fit(self):
        training_data = self.factorization_machine.training_data
        
        loss_data = []
        
        
        for i in range(self.iterations):
            print("iter %d" % i)
            bias_gradient = 0
            w_gradient = np.zeros_like(self.factorization_machine.w_pars)
            v_gradient = np.zeros_like(self.factorization_machine.v_matrix)
        
        
            for row_idx in range(training_data.shape[0]):
                current_bias_gradient, current_pars_gradient, current_v_gradient = self.factorization_machine.loss_MSE_gradient(training_data[row_idx, :])

                bias_gradient += current_bias_gradient
                w_gradient += current_pars_gradient
                v_gradient += current_v_gradient
            
            
            loss_MSE = self.factorization_machine.loss_MSE()
            bias_gradient *= loss_MSE
            w_gradient *= loss_MSE
            v_gradient *= loss_MSE
            
            """
            print("b grad", bias_gradient)
            print("w grad", w_gradient)
            print("v grad", v_gradient)
            """
            
            """
            print("b", self.factorization_machine.w_bias)
            print("w", self.factorization_machine.w_pars)
            print("v", self.factorization_machine.v_matrix)
            """
            
            
            print("loss MSE", loss_MSE)
            loss_data.append(loss_MSE)
            
            self.factorization_machine.w_bias -= self.learning_rate * bias_gradient
            self.factorization_machine.w_pars -= self.learning_rate * w_gradient
            self.factorization_machine.v_matrix -= self.learning_rate * v_gradient
            
            
        return loss_data
            
        