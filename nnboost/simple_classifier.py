import tensorflow as tf
import numpy as np

class SimpleClassifier():
  def __init__(self, activation='softmax', loss='categorical_crossentropy', seed=None):
    self.__seed = seed
    self.__model = None
    self.__activation = activation
    self.__loss = loss


  def fit(self, X, y, epochs=1000, verbose=0):
    tf.random.set_seed(self.__seed)

    x_width = X.shape[1]
    
    #EarlyStop
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                patience=10, 
                                                verbose=0)
    
    input_ = tf.keras.layers.Input(shape=[x_width])
    hidden1 = tf.keras.layers.Dense(2*x_width, activation="relu")(input_)
    hidden2 = tf.keras.layers.Dense(x_width, activation="relu")(hidden1)
    hidden3 = tf.keras.layers.Dense(x_width // 2, activation="relu")(hidden2)
    concat = tf.keras.layers.concatenate([input_, hidden3])
    output = tf.keras.layers.Dense(y.shape[1], activation=self.__activation, name="output")(concat)
    self.__model = tf.keras.Model(inputs=[input_], outputs=[output])

    if self.__loss in ['mae','mse','MAPE']:
      self.__model.compile(loss=self.__loss, optimizer=tf.optimizers.Adam(lr=0.001, decay=1e-6), metrics=[self.__loss])
    else:
      self.__model.compile(loss=self.__loss, optimizer=tf.optimizers.Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
    self.__model.fit(X, y, epochs=epochs, callbacks=[callback], verbose=verbose)
    
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