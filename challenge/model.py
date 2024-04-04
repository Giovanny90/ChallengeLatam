import pandas as pd
import xgboost as xgb
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from datetime import datetime
from fastapi import FastAPI, HTTPException
import numpy as np
import os
class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self.model_path="model.json"
        self.top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self.caracteristicas=['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada',
        'OPERA_Air France', 'OPERA_Alitalia', 'OPERA_American Airlines',
        'OPERA_Austral', 'OPERA_Avianca', 'OPERA_British Airways',
        'OPERA_Copa Air', 'OPERA_Delta Air', 'OPERA_Gol Trans',
        'OPERA_Grupo LATAM', 'OPERA_Iberia', 'OPERA_JetSmart SPA',
        'OPERA_K.L.M.', 'OPERA_Lacsa', 'OPERA_Latin American Wings',
        'OPERA_Oceanair Linhas Aereas', 'OPERA_Plus Ultra Lineas Aereas',
        'OPERA_Qantas Airways', 'OPERA_Sky Airline', 'OPERA_United Airlines',
        'TIPOVUELO_I', 'TIPOVUELO_N', 'MES_1', 'MES_2', 'MES_3', 'MES_4',
        'MES_5', 'MES_6', 'MES_7', 'MES_8', 'MES_9', 'MES_10', 'MES_11',
        'MES_12']
    
    def initialize_model(self) -> None:
        if self._model is None:
            if os.path.exists(self.model_path):
                self._model = xgb.Booster()
                self._model.load_model(self.model_path)
            else:
                raise FileNotFoundError(f"Model file '{self.model_path}' not found.")

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1)
        column_set = set(features.columns)
        data_set = set(self.caracteristicas)
        top_10_features_set= set(self.top_10_features)
        if target_column:
            if column_set.issubset(data_set):
                features = features.reindex(columns=self.top_10_features, fill_value=False)
                features=features[self.top_10_features]
                data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
                data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')
                min_diff = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
                data['min_diff'] =min_diff
                threshold_in_minutes = 15
                data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
                target = pd.DataFrame(data['delay'])
            else:
                raise HTTPException(status_code=400, detail="No se proporcionaron datos de vuelo")          
            target = pd.DataFrame(data[target_column])
            return features, target
        else:
            print(column_set)
            if not column_set.issubset(data_set):
                raise HTTPException(status_code=400, detail="No se proporcionaron datos de vuelo") 
            features 
            features = features.reindex(columns=self.top_10_features, fill_value=False)
            return features

        

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Entrenar el modelo XGBoost
        if target is not None and features is not None:
            x_train, x_test, y_train, y_test = train_test_split(features[self.top_10_features], target, test_size = 0.33, random_state = 42)
            n_y0 = len(y_train[y_train["delay"] == 0])
            n_y1 = len(y_train[y_train["delay"] == 1])
            scale = n_y0/n_y1
            if self._model is None:
                self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
            self._model.fit(x_train, y_train)
            self._model.save_model("model.json")
        return self._model 

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        if self._model is None:
            self.initialize_model()

        column_set = set(features.columns)
        if column_set.issubset(self.top_10_features):
            dmatrix = xgb.DMatrix(data=features)
            predictions_prob=self._model.predict(dmatrix)
            predictions = [int(round(prob)) for prob in predictions_prob]
        else:
            predictions=[0]
        return predictions