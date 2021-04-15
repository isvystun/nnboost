import tensorflow as tf
import numpy as np

# local
from .simple_classifier import SimpleClassifier

class NNBoostClassifier:
  def __init__(self, *, n_estimators=100, degree=1, learning_rate=1, seed=None):
    self.__seed : int = seed
    self.__n_estimators : int = n_estimators # number of neural networks
    self.__estimators : dict[int, SimpleRegressor] = {}
    self.__learning_rate : float = learning_rate
    self.__metrics = {}
    self.__degree = degree

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
    
    predictor = None
    
    height = original_output.shape[0]
    width =  len(np.unique(original_output))
    self.__base = np.full((height, width), 1/width) 

    self.__gamma = {}
    updated_model=self.__base
    for i in range(self.__n_estimators):
      output = original_output - updated_model
      self.__estimators[i] = SimpleClassifier().fit(input, output, verbose=verbose)
      
      predictor = self.__estimators[i].predict(input)
#       self.__gamma[i] = np.mean((original_output - updated_model)/predictor)
      self.__gamma[i] = np.linalg.pinv(predictor.reshape(-1,1)).dot(original_output - updated_model)[0]
      
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
                          for i in self.__estimators.keys()
                  ], axis=0) + self.__base 


    
  def save_model(self, file_name):
    pass


  def load_model(self, file_name):
    pass