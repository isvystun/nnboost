import tensorflow as tf
import numpy as np



# local
from .simple_classifier import SimpleClassifier

class NNBoostClassifier:
  def __init__(self, *, n_estimators=100, learning_rate=1, bias=False, seed=None):
    self.__n_estimators : int = n_estimators # number of neural networks
    self.__estimators : dict[int, SimpleClassifier] = {}
    self.__learning_rate : float = learning_rate
    self.__metrics = {}
    self.predictor = {}
    self.__bias = bias

  @property
  def estimators(self):
    return self.__estimators

  @property
  def metrics(self):
    return self.__metrics


  def fit(self, X, y, *, verbose=0):
    
    unique = np.unique(y).shape[0]
    original_output = tf.one_hot(y, unique)
    original_output = np.where(original_output==0, -1, original_output)
    if self.__bias:
      _X = np.c_[np.ones((X.shape[0],1)), X.copy()]
    else:
      _X = X.copy()

    predictor = None
    
  
    # self.__base = SimpleClassifier(activation='linear',loss='mse').fit(_X, original_output)
    self.__base = np.mean(original_output[0])
    self.__gamma = {}
    # updated_model=self.__base.predict_proba(_X)
    updated_model=np.mean(original_output[0])
    for i in range(self.__n_estimators):
      output = original_output - updated_model

      self.__estimators[i] = SimpleClassifier(activation='linear',loss='mse').fit(_X, output, verbose=verbose)
      
      predictor = self.__estimators[i].predict_proba(_X)
      self.__gamma[i] = np.linalg.pinv(predictor).dot(output)[0]
      # self.__gamma[i] = 1
      self.predictor[i] = predictor
      print(f'{np.sum(predictor)}')
      updated_model += predictor*self.__gamma[i]*self.__learning_rate
        
      mae = self.__estimators[i].metrics[1].result().numpy()
      self.__metrics[i] = mae
      print(f"NN #{i} is done -> MAE : {mae}")
    return self


  def predict_proba(self, X):
    if self.__bias:
      _X = np.c_[np.ones((X.shape[0],1)), X.copy()]
    else:
      _X = X.copy()
    return np.sum([self.__estimators[i].predict_proba(_X)*self.__learning_rate*self.__gamma[i]
                          for i in self.__estimators.keys()
                  ], axis=0) + self.__base #.predict_proba(_X) 


  def predict(self, X):
      return np.argmax(self.predict_proba(X), axis=1)

  
  def __normalize_zero(self, y):
    n = y.shape[1]
    return y - np.sum(y, axis=1).reshape(-1,1)/n


  def save_model(self, file_name):
    pass


  def load_model(self, file_name):
    pass