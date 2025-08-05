import numpy as np

class MLP:
    def __init__(self, layer_dims, activations, learning_rate):
        self.parameters = {}
        self.layer_dims = layer_dims
        self.activations = activations
        self.grads = {}
        self.caches = None
        self.layers = len(self.layer_dims) - 1
        self.learning_rate = learning_rate

    def initialize_parameters(self,seed = 32):
        np.random.seed(seed)

        for l in range(1, self.layers+1):
            self.parameters["W" + str(l)] = np.random.randn(self.layer_dims[l],self.layer_dims[l-1]) * 0.01
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l],1))

    
    def linear_forward(self, A, W, b):
        linear_cache = (A, W, b)
        Z = np.dot(W, A) + b
        return Z, linear_cache

    def activation_forward(self, Z, activation):
        if activation == "relu":
            A = np.maximum(0,Z)
        elif activation == "sigmoid": 
            A = 1/(1+np.exp(-Z))   
        elif activation == "softmax":
            exp_shift = np.exp(Z)
            A = exp_shift / np.sum(exp_shift, axis=0, keepdims=True)
        elif activation == "linear":
            A = Z   
        activation_cache = Z     
        return A, activation_cache

    def forward_prop(self, X):
        caches = []
        A_prev = X

        for l in range(1, self.layers+1):
            W = self.parameters["W" + str(l)]
            b = self.parameters["b" + str(l)]
            activation = self.activations[l-1]    

            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.activation_forward(Z, activation)
            A_prev = A
            caches.append((linear_cache, activation_cache))
        self.caches = caches
        return A_prev

    def compute_cost(self, AL, Y):
        m = AL.shape[1] 
        cost = 0

        if self.activations[-1] == "linear":
            #MSE- Mean Squared Error
            cost = np.sum(np.square(Y-AL))/(2*m)
        elif self.activations[-1] == "sigmoid":
            #Binary cross-entropy
            cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))/m
        elif self.activations[-1] == "softmax":
            #Categorical cross-entropy
            cost = -np.sum(np.log(Y))/m
        return cost

    def linear_backward(self, dZ, linear_cache):
        A, W, b = linear_cache
        m = dZ.shape[1]
        dW = np.dot(dZ, np.transpose(A))/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(np.transpose(W),dZ)
        return dA_prev, dW, db
 

    def activation_backward(self, dA, activation_cache, activation):
        Z = activation_cache
        
        if activation == "relu":
            dZ = dA * (Z > 0)
        elif activation == "sigmoid": 
            S = 1 / (1 + np.exp(-Z))
            dZ = dA * S * (1 - S) 
        elif activation == "linear":
            dZ = dA 
        return dZ

    def output_backward(self, AL, Y):
        m = AL.shape[1] 
        dZ_l = (AL - Y)/m 
        return dZ_l


    
    def backward_prop(self, AL, Y):

        self.grads = {}

        linear_cache, activation_cache = self.caches[-1]

        dZ_l = self.output_backward(AL, Y)
        dA_prev, dW, db = self.linear_backward(dZ_l, linear_cache)

        self.grads["dW" + str(self.layers)] = dW
        self.grads["db" + str(self.layers)] = db        



        for l in range(self.layers-1, 0, -1):
            activation = self.activations[l-1]
            linear_cache, activation_cache = self.caches[l-1]

            dZ = self.activation_backward(dA_prev, activation_cache, activation)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

            self.grads["dW" + str(l)] = dW
            self.grads["db" + str(l)] = db




if __name__ == "__main__":
    # mlp = MLP([2,3,5], ["relu","relu","softmax"], 0.01)
    # mlp.initialize_parameters()
    # print(mlp.parameters)

    layer_dims   = [2, 3, 5]                     # 2 wejścia → 1 warstwa ukryta (3) → warstwa wyjściowa (5)
    activations  = ["relu", "relu", "softmax"]
    learning_rate = 0.01
    seed = 42


    mlp = MLP(layer_dims, activations, learning_rate)
    mlp.initialize_parameters(seed=seed)


    np.random.seed(123)
    X = np.random.randn(layer_dims[0], 10)  #(2,10)

    AL = mlp.forward_prop(X)

    print("Wejście X.shape =", X.shape)
    print("Wyjście AL.shape =", AL.shape)
    print("AL:\n", AL)






