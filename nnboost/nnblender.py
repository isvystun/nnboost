from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np

#local
from .neuron import Neuron
from .nn_boost_regressor import NNBoostRegressor


class NNBlender:
    def __init__(self):
        self.is_trained = False

    def fit(self, X, y):
        self.__num_col = list(X.select_dtypes(include=['number']).columns)
        self.__cat_col = list(set(X.columns) - set(self.__num_col))

        self.__num_transformer = Pipeline(steps=[
                ('numerical_imputer', SimpleImputer(strategy='median')),
                ('numerical_scaler', StandardScaler())
        ])
        
        self.__X_num = self.__num_transformer.fit_transform(X[self.__num_col])
        
        self.__X_cat = [Pipeline(steps=[('categorical_imputer', SimpleImputer(fill_value='NA', strategy='constant')),
                                 ('onehot_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])
                        for n in self.__cat_col
                       ]

        self.__y_transformer = MinMaxScaler()
        _y = self.__y_transformer.fit_transform(y.to_numpy().reshape(-1,1)).flatten()
        
        # _y = y        


        self.__num_models = [Neuron(ones_column=True).fit(self.__X_num[:,i].reshape(-1,1), _y) for i in range(self.__X_num.shape[1])]
        
        self.__cat_models = [Neuron().fit(x.fit_transform(X[[ self.__cat_col[i] ]]), _y) for i,x in enumerate(self.__X_cat)]

        self.num_models_features = [m.predict(self.__X_num[:, i].reshape(-1,1)) for i, m in enumerate(self.__num_models)]
        self.cat_models_features = [m.predict(self.__X_cat[i].transform(X[[ self.__cat_col[i] ]])) for i, m in enumerate(self.__cat_models)]


        self.features = np.concatenate(self.num_models_features + self.cat_models_features, axis=1)
        self.gamma = np.array([np.linalg.pinv(self.features[:, c].reshape(-1,1)).dot(_y)[0]
                        for c in range(self.features.shape[1])
                     ])
        print(self.gamma)
        self.__nnboost = NNBoostRegressor(n_estimators=100, learning_rate=0.2).fit(self.gamma*self.features, _y)
        # self.__nnboost = Neuron(activation='linear').fit(self.gamma*self.features, _y)
        self.is_trained = True
        return self


    def predict(self, X):
        X_num = self.__num_transformer.transform(X[self.__num_col])

        num_models_features = [m.predict(X_num[:, i].reshape(-1,1)) for i, m in enumerate(self.__num_models)]
        cat_models_features = [m.predict(self.__X_cat[i].transform(X[[ self.__cat_col[i] ]])) for i, m in enumerate(self.__cat_models)]
        features = np.concatenate(num_models_features + cat_models_features, axis=1)

        pred = self.__nnboost.predict(self.gamma*features)
        pred_inv = self.__y_transformer.inverse_transform(pred.reshape(-1,1)).flatten()
        return pred_inv