import tensorflow as tf

class SimpleRegressor():

  def __init__(self,*, seed=None):
    self.__seed = seed
    self.__model = None


  def fit(self, X, y,*, loss='mae', epochs=5000, verbose=0):
    output = X.shape[1]   
    tf.random.set_seed(self.__seed)
    
    #EarlyStop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                patience=4, 
                                                verbose=0)
    kernel_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.2)
    bias_init = tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.02)
    
    input_ = tf.keras.layers.Input(shape=[output])
    hidden1 = tf.keras.layers.Dense(2*output, activation="relu")(input_)
    hidden2 = tf.keras.layers.Dense(output, activation="relu")(hidden1)
    hidden3 = tf.keras.layers.Dense(output // 2, activation="relu")(hidden2)
    concat = tf.keras.layers.concatenate([input_, hidden3])
    output = tf.keras.layers.Dense(1, name="output")(concat)
    self.__model = tf.keras.Model(inputs=[input_], outputs=[output])
    
    
    # self.__model = tf.keras.Sequential([
    #   tf.keras.layers.Dense(output*2, 
    #                         activation='relu',
    #                         kernel_initializer=kernel_init, 
    #                         bias_initializer= bias_init
    #   ),
    #   tf.keras.layers.Dense(output, 
    #                         activation='relu',
    #                         kernel_initializer=kernel_init, 
    #                         bias_initializer=bias_init
    #   ),
    #   tf.keras.layers.Dense(1,
    #                         kernel_initializer=kernel_init, 
    #                         bias_initializer=bias_init
    #   )
    # ])
    self.__model.compile(loss=loss, optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=[loss])
    self.__model.fit(X, y, epochs=epochs, callbacks=[callback], verbose=verbose)
    
    return self


  def predict(self, X):
    return self.__model.predict(X).flatten()

  def evaluate(self, X, y):
    self.__model.evaluate(X, y)

  @property
  def metrics(self):
    return self.__model.metrics