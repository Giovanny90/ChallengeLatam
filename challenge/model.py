import pandas as pd
import xgboost as xgb
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from datetime import datetime
from fastapi import FastAPI, HTTPException
import numpy as np
class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

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
        top_10_features = [
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

        caracteristicas=['OPERA_Aerolineas Argentinas', 'OPERA_Aeromexico', 'OPERA_Air Canada',
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
        if target_column:
            if target_column is not None:
                if target_column not in data.columns:
                    features = pd.concat([
                    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                    pd.get_dummies(data['MES'], prefix = 'MES')], 
                    axis = 1)
                    column_set = set(features.columns)
                    data_set = set(caracteristicas)
                    if column_set.issubset(data_set):
                        features=features[top_10_features]
                        #features=features
                        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
                        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')
                        min_diff = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
                        data['min_diff'] =min_diff
                        threshold_in_minutes = 15
                        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
                        target = pd.DataFrame(data[target_column])
                    else:
                        raise HTTPException(status_code=400, detail="No se proporcionaron datos de vuelo")
                else:
                    features = data.drop(columns=[target_column], errors='ignore')
                    features = pd.concat([
                    pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                    pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                    pd.get_dummies(data['MES'], prefix = 'MES')], 
                    axis = 1)
                    features=features[top_10_features]
                    #features=features
                    data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
                    data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')
                    min_diff = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60
                    data['min_diff'] =min_diff
                    threshold_in_minutes = 15
                    data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
                    target = data['delay']
                return features,target
        else:
            features = pd.concat([
                        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                        pd.get_dummies(data['MES'], prefix = 'MES')], 
                        axis = 1)
            column_set = set(features.columns)
            data_set = set(caracteristicas)
            top_10_features_set = set(top_10_features)
            if column_set.issubset(data_set):
               if column_set.issubset(top_10_features):
                features = features.reindex(columns=top_10_features, fill_value=False)
                features=features[top_10_features]
               else:
                features = features.reindex(columns=top_10_features, fill_value=False)
                features=features
            else:
                    raise HTTPException(status_code=400, detail="No se proporcionaron datos de vuelo para entrenar el modelo")
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
        top_10_features = [
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
        if target is not None and features is not None:
            x_train, x_test, y_train, y_test = train_test_split(features[top_10_features], target, test_size = 0.33, random_state = 42)
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
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        top_10_features = [
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

        if self._model is not None:
            print("PRUEEEEEEBAAAA")
            column_set = set(features.columns)
            top_10_features = set(top_10_features)

            print(column_set)
            if column_set.issubset(top_10_features):
                predictions_prob =self._model.predict(features)
                predictions = [int(round(prob)) for prob in predictions_prob]
            else:
                predictions=[0]
            
        else:
            column_set = set(features.columns)
            top_10_features = set(top_10_features)
            if column_set.issubset(top_10_features):
                model_xgb = xgb.Booster()
                model_xgb.load_model("model.json")
                dmatrix = xgb.DMatrix(data=features)
                predictions_prob=model_xgb.predict(dmatrix)
                print(predictions_prob)
                #print(predictions_prob)
                predictions = [int(round(prob)) for prob in predictions_prob]
                print(predictions)
            else:
                predictions=[0]
        return predictions