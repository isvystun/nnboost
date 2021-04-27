from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
import numpy as np

class Neuron():
    def __init__(self, activation='sigmoid', ones_column=False):
        self.__ones_column = ones_column
        kernel_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.1)
        bias_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.001)
        self.__model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, 
                                  activation=activation, 
                                  kernel_initializer=kernel_init, 
                                  bias_initializer=bias_init)
        ])

        self.__model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.05), 
                            loss='mse', 
                            metrics='mse')

    # X must be normalized
    def fit(self, X, y):
        if X.ndim != 2:
            raise Exception('X must be 2 dimensional array')
   
        self.__X = np.c_[np.ones((X.shape[0],1)), X] if self.__ones_column else X

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                    patience=4, 
                                                    verbose=0)

        self.__model.fit(self.__X, y, epochs=100, callbacks=[callback], verbose=0)

        return self


    def predict(self, X):
        if X.ndim != 2:
            raise Exception('X must be 2 dimensional array')
        
        self.__X = np.c_[np.ones((X.shape[0],1)), X] if self.__ones_column else X
        
        return self.__model.predict(self.__X)
    
