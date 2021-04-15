import tensorflow as tf
import numpy as np

# local
from .simple_regressor import SimpleRegressor

class NNBoostRegressor:
  def __init__(self, *, n_estimators=100, degree=1,base_model_method='mean', learning_rate=1, seed=None):
    self.__seed : int = seed
    self.__n_estimators : int = n_estimators # number of neural networks
    self.__estimators : dict[int, SimpleRegressor] = {}
    self.__learning_rate : float = learning_rate
    self.__metrics = {}
    self.__degree = degree
    self.__base_model_method = base_model_method

  @property
  def estimators(self):
    return self.__estimators

  @property
  def metrics(self):
    return self.__metrics


  def fit(self, X, y, *, verbose=0):
    concat = []#np.ones((X.shape[0],1))]
    for d in range(1, self.__degree+1):
      concat.append(X.copy()**d)
    
    input = np.concatenate(concat, axis=1)
    original_output = y.copy()
    
#     if (isinstance(self.__base_model_method, str) and self.__base_model_method in ['mean', 'median']):
#         base_model = getattr(np, self.__base_model_method)
#         self.__base = base_model(original_output)
#     elif type(self.__base_model_method) in [int, float]:
#         self.__base = self.__base_model_method
#     else:
#         raise Exception("base_model_method must be ['mean', 'median'] or any int or float number.")
    self.__base = SimpleRegressor().fit(input, original_output, verbose=verbose)
    self.__gamma = {}
    updated_model=self.__base.predict(input)
#     updated_model=self.__base
    for i in range(self.__n_estimators):
      output = original_output - updated_model
      self.__estimators[i] = SimpleRegressor().fit(input, output, verbose=verbose)
      
      predictor = self.__estimators[i].predict(input)
#       self.__gamma[i] = np.median((original_output - updated_model)/predictor)
#       self.__gamma[i] = np.linalg.pinv(predictor.reshape(-1,1)).dot(original_output - updated_model)[0]
#       self.__gamma[i] = 1
      gamma = self.__newton(original_output, updated_model, predictor)
      if np.isnan(gamma):
            break
      self.__gamma[i] = gamma
      print(f"gamma -> {gamma}")
      print(f"gamma2 -> {np.linalg.pinv(predictor.reshape(-1,1)).dot(original_output - updated_model)[0]}")
      updated_model += predictor*self.__gamma[i]*self.__learning_rate
        
      mae = self.__estimators[i].metrics[0].result().numpy()
      self.__metrics[i] = mae
      print(f"NN #{i} is done -> MAE : {mae}")
    return self



  def predict(self, X):
    input_poly = []#np.ones((X.shape[0],1))]
    for d in range(1, self.__degree+1):
      input_poly.append(X.copy()**d)

    _X = np.concatenate(input_poly, axis=1)

    return np.sum([self.__estimators[i].predict(_X)*self.__learning_rate*self.__gamma[i]
                          for i in self.__gamma.keys()
                  ], axis=0) + self.__base.predict(_X) 

#     return np.sum([self.__estimators[i].predict(_X)*self.__learning_rate*self.__gamma[i]
#                       for i in self.__estimators.keys()
#               ], axis=0) + self.__base
  
    
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