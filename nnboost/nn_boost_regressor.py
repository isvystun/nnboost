import tensorflow as tf
import numpy as np

import tensorflow.keras.metrics as tm

# local
from .simple_regressor import SimpleRegressor
from .simple_classifier import SimpleClassifier

from .autoencoder import AutoEncoder

class NNBoostRegressor:
  def __init__(self, *, n_estimators=100, base_model_method='mean', learning_rate=1, auto_encoder=False, chasing_sign_vector=False, seed=None):
    self.__seed : int = seed
    self.__n_estimators : int = n_estimators # number of neural networks
    self.__estimators : dict[int, SimpleRegressor] = {}
    self.__learning_rate : float = learning_rate
    self.__metrics = {}
    self.__base_model_method = base_model_method
    self.__auto_encoder = auto_encoder
    self.__is_trained = False
    self.__chasing_sign_vector = chasing_sign_vector
    self.__gamma_model = {}


  @property
  def estimators(self):
    return self.__estimators

  @property
  def metrics(self):
    return self.__metrics


  def fit(self, X, y, *, verbose=0):
    x_width = X.shape[1]
    # concat = []#np.ones((X.shape[0],1))]
    self._X_train = X.copy()

    self.__original_output = y.copy()
    

    if self.__auto_encoder:
      self.auto = AutoEncoder(x_width)
      self.auto.compile(loss='mse', 
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6), 
                   metrics=['mae'])
      
      callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                            patience=3, 
                                            verbose=0)
      
      self.auto.fit(x=X.copy(), 
               y=X.copy(), 
               epochs=100, 
               batch_size=32, 
               callbacks=[callback],
               verbose=0)

      self._X_train = self.auto.encoder.predict(X.copy())
    if (isinstance(self.__base_model_method, str) and self.__base_model_method in ['mean', 'median']):
        base_model = getattr(np, self.__base_model_method)
        self.__base = base_model(self.__original_output)
    elif type(self.__base_model_method) in [int, float]:
        self.__base = self.__base_model_method
    else:
        raise Exception("base_model_method must be ['mean', 'median'] or any int or float number.")
    # self.__base = SimpleRegressor().fit(self._X_train, self.__original_output, verbose=verbose)
    self.__delta = {}
    # updated_model=self.__base.predict(self._X_train)
    updated_model=self.__base

    for i in range(self.__n_estimators):
      output = self.__original_output - updated_model
      
      if self.__chasing_sign_vector:
        classifier_output = output.copy()
        threshhold = 0.01 #1%

        perc = classifier_output / self.__original_output

        self.__delta[i] = {}

        self.__delta[i][-1] = np.nan_to_num(np.median(classifier_output[perc<=-threshhold]))
        self.__delta[i][0] = np.nan_to_num(np.median(classifier_output[(perc<threshhold) & (perc>-threshhold)]))
        self.__delta[i][1] = np.nan_to_num(np.median(classifier_output[perc>=threshhold]))

        classifier_output[perc<=-threshhold] = 0 # (-1) negative direction
        classifier_output[(perc<threshhold) & (perc>-threshhold)] = 1 # 0
        classifier_output[perc>=threshhold] = 2 # (1) positive direction

        classifier_output = tf.one_hot(classifier_output, 3)
        self.__estimators[i] = SimpleClassifier().fit(self._X_train, classifier_output, verbose=verbose)
      
        predictor_sign = self.__estimators[i].predict(self._X_train) - 1
        
        sign_to_delta = lambda x : self.__delta[i][x]
        vfunc = np.vectorize(sign_to_delta)
        predictor = vfunc(predictor_sign)
      else:
        self.__estimators[i] = SimpleRegressor().fit(self._X_train, output, verbose=verbose)
      
        predictor = self.__estimators[i].predict(self._X_train)


      updated_model += predictor * self.__learning_rate

      print(f"MAPE : {np.mean(np.abs(self.__original_output - updated_model)*100/self.__original_output)} %")  
      
      mae = self.__estimators[i].metrics[0].result().numpy()
      self.__metrics[i] = mae
      print(f"NN #{i} is done -> MAE : {mae}")
    
    self.__is_trained = True
    return self


  def predict(self, X):
    if not self.__is_trained:
      raise Exception('Model must be trained first. Please use model.fit(...) method.')
    
    if self.__auto_encoder:
      _X = self.auto.encoder.predict(X.copy())
    else:
      _X = X.copy()

    if self.__chasing_sign_vector:
      p_delta = []
      for i in self.__estimators.keys():
        p_sign = self.__estimators[i].predict(_X) - 1
        sign_to_delta = lambda x : self.__delta[i][x]
        vfunc = np.vectorize(sign_to_delta)
        p_delta.append(vfunc(p_sign))

      return np.sum(p_delta, axis=0)*self.__learning_rate + self.__base
    else:
      return np.sum([self.__estimators[i].predict(_X)*self.__learning_rate
                        for i in self.__estimators.keys()
                ], axis=0) + self.__base
  
  
  def metric(self, metric='mae'):
    m = getattr(tm, metric)
    return m(self.__original_output, self.predict(self._X_train)).numpy()
    

  def __newton(self, y, F, h):
    return np.sum(y - F) / np.sum(h)



  def __dL(self, y, y_pred, lr):
    return -2*lr*np.sum(y-y_pred)



  def __gradient(self, y, F, h, lr=0.1, eps=0.01):
    gamma_prev, gamma_next = 1, 0
    m = y.shape[0]
    print(f"m = {m}")
    print(f"gradient - {h.mean()}")

    for _ in range(100):
      gamma_next = gamma_prev - lr * 2/m * h.dot(h*gamma_prev - y + F)
      
      if np.abs(gamma_next - gamma_prev) < eps:
        break
    
    return gamma_next


# 
  def save_model(self, file_name):
    pass


  def load_model(self, file_name):
    pass