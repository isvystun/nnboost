import tensorflow as tf
import numpy as np

import tensorflow.keras.metrics as tm
# local
from .simple_regressor import SimpleRegressor
from .autoencoder import AutoEncoder

class NNBoostRegressor:
  def __init__(self, *, n_estimators=100,base_model_method='mean', learning_rate=1, auto_encoder=False, seed=None):
    self.__seed : int = seed
    self.__n_estimators : int = n_estimators # number of neural networks
    self.__estimators : dict[int, SimpleRegressor] = {}
    self.__learning_rate : float = learning_rate
    self.__metrics = {}
    self.__base_model_method = base_model_method
    self.__auto_encoder = auto_encoder
    self.__is_trained = False


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
#     if (isinstance(self.__base_model_method, str) and self.__base_model_method in ['mean', 'median']):
#         base_model = getattr(np, self.__base_model_method)
#         self.__base = base_model(self.__original_output)
#     elif type(self.__base_model_method) in [int, float]:
#         self.__base = self.__base_model_method
#     else:
#         raise Exception("base_model_method must be ['mean', 'median'] or any int or float number.")
    self.__base = SimpleRegressor(output_activation='softplus').fit(self._X_train, self.__original_output, verbose=verbose)
    self.__gamma = {}
    updated_model=self.__base.predict(self._X_train)
#     updated_model=self.__base
    for i in range(self.__n_estimators):
      output = self.__original_output - updated_model
      self.__estimators[i] = SimpleRegressor().fit(self._X_train, output, verbose=verbose)
      
      predictor = self.__estimators[i].predict(self._X_train)
#       self.__gamma[i] = np.median((self.__original_output - updated_model)/predictor)
#       self.__gamma[i] = np.linalg.pinv(predictor.reshape(-1,1)).dot(self.__original_output - updated_model)[0]
      
      gamma = self.__newton(self.__original_output, updated_model, predictor)
      # gamma = 1
      if np.isnan(gamma):
            break
      self.__gamma[i] = gamma
      updated_model += predictor*self.__gamma[i]*self.__learning_rate
        
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

    return np.sum([self.__estimators[i].predict(_X)*self.__gamma[i]*self.__learning_rate
                          for i in self.__gamma.keys()
                  ], axis=0) + self.__base.predict(_X) 

#     return np.sum([self.__estimators[i].predict(_X)*self.__learning_rate*self.__gamma[i]
#                       for i in self.__estimators.keys()
#               ], axis=0) + self.__base
  
  def metric(self, metric='mae'):
    m = getattr(tm, metric)
    return m(self.__original_output, self.predict(self._X_train)).numpy()
    

  def __newton(self, y, F, h):
    gamma_prev = 1
    gamma_next = 0
    for _ in range(1000):
      L = np.sum((y-(F+h*gamma_prev))**2)
      dL = 2*(np.sum(h)*np.sum(h*gamma_prev+F-y))
      gamma_next = gamma_prev - L/ dL
      if abs(gamma_next-gamma_prev) < 0.1:
        return gamma_next
      else:
        gamma_prev = gamma_next
    return np.nan


# 
  def save_model(self, file_name):
    pass


  def load_model(self, file_name):
    pass