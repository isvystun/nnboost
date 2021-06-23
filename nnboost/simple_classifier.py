import tensorflow as tf
import numpy as np

class SimpleClassifier():
  def __init__(self, activation='softmax', loss='categorical_crossentropy', seed=None):
    self.__seed = seed
    self.__model = None
    self.__activation = activation
    self.__loss = loss


  def fit(self, X, y, sample_weight=None, epochs=1000, verbose=0):
    tf.random.set_seed(self.__seed)
    
    if sample_weight is None:
      sample_weight = np.ones(shape=(len(y),))
    
    x_width = X.shape[1]
    
    #EarlyStop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                patience=10, 
                                                verbose=0)
    
    input_ = tf.keras.layers.Input(shape=[x_width])
    hidden1 = tf.keras.layers.Dense(2*x_width, activation="relu")(input_)
    hidden2 = tf.keras.layers.Dense(x_width, activation="relu")(hidden1)
    output = tf.keras.layers.Dense(y.shape[1], activation=self.__activation, name="output")(hidden2)
    self.__model = tf.keras.Model(inputs=[input_], outputs=[output])

    if self.__loss in ['mae','mse','MAPE']:
      self.__model.compile(loss=self.__loss, optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=[self.__loss])
    else:
      self.__model.compile(loss=self.__loss, optimizer=tf.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    self.__model.fit(X, y, sample_weight=sample_weight, epochs=epochs, callbacks=[callback], verbose=verbose)
    
    return self


  def predict(self, X):
    return np.argmax(self.predict_proba(X), axis=1)


  def predict_proba(self, X):
    return self.__model.predict(X)


  # def evaluate(self, X, y):
  #   self.__model.evaluate(X, y)

  @property
  def metrics(self):
    return self.__model.metrics