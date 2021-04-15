import tensorflow as tf

class SimpleClassifier():
  def __init__(self, seed=None):
    self.__seed = seed
    self.__model = None


  def fit(self, X, y, epochs=1000, verbose=0):
    output = len(np.unique(y))   
    tf.random.set_seed(self.__seed)
    
    #EarlyStop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                patience=4, 
                                                verbose=0)
    kernel_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=1)
    bias_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.05)
    
    self.model = tf.keras.Sequential([
      tf.keras.layers.Dense(output*2, 
                            activation='relu',   
                            kernel_initializer=kernel_init, 
                            bias_initializer=bias_init
      ),
      tf.keras.layers.Dense(output, 
                            activation='relu',
                            kernel_initializer=kernel_init, 
                            bias_initializer=bias_init
      ),
      tf.keras.layers.Dense(output, 
                            activation='softmax',
                            kernel_initializer=kernel_init, 
                            bias_initializer=bias_init
      ),
    ])
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    self.model.fit(X, y, epochs=epochs, callbacks=[callback], verbose=verbose)
    
    return self


  def predict(self, X):
    return self.model.predict(X)


  def evaluate(self, X, y):
    self.model.evaluate(X, y)