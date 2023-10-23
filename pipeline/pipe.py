import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from lightgbm import Booster
from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap
import time


class Pipeline:

    def __init__(self) -> None:

        self.cb_clf = CatBoostClassifier().load_model('./saved_models/cb_clf')
        self.lgbm_clf = Booster(model_file='./saved_models/lgbm_clf')
        self.xgb_clf = XGBClassifier()
        self.xgb_clf.load_model('./saved_models/xgb_clf')
        self.tabnet_clf = TabNetClassifier()
        self.tabnet_clf.load_model('./saved_models/tabnet_clf.zip')
        self.meta_clf = CatBoostClassifier().load_model('./saved_models/meta_clf')

        self.cat_features = ['provider', 'material', 'category_manager', 'operation_manager', 'factory', 'purchase_org', 
                'purchase_group', 'balance_unit', 'unit', 'material_group', 'supply_variant', 'mon1', 'mon2', 'mon3', 'weekday2']

    def predict(self, csv_path:str, save_dir: str='./preds') -> None:
        """Prediction and saving the results"""

        self.df = self.__readset__(csv_path=csv_path)

        df = self.df.copy()
        df['cb_clf'] = self.cb_clf.predict(self.df)
        df['lgbm_clf'] = self.lgbm_clf.predict(self.df)
        df['xgb_clf'] = self.xgb_clf.predict(self.df)
        df['tabnet_probas_0'] = list(map(lambda x : x[0], self.tabnet_clf.predict_proba(self.df.values)))
        df['tabnet_probas_1'] = list(map(lambda x : x[1], self.tabnet_clf.predict_proba(self.df.values)))

        # probas and preds
        preds = self.meta_clf.predict(df)
        probas = self.meta_clf.predict_proba(df)

        res = self.df.copy()
        res['preds'] = preds
        res['0_probas'] = [x[0] for x in probas]
        res['1_probas'] = [x[1] for x in probas]

        feature_importances = self.cb_clf.get_feature_importance(Pool(df, cat_features=self.cat_features), prettified=True)

        # shap
        if len(self.df) > 300:
            l = 300
        else:
            l = len(self.df)
        explainer = shap.Explainer(self.cb_clf)
        shap_values = explainer.shap_values(self.df.loc[:l])

        #shap.decision_plot(explainer.expected_value, shap_values, self.df.columns)
        shap.decision_plot(explainer.expected_value, shap_values, self.df.columns)



        
        #shap.summary_plot(shap_values, self.df.loc[:l], show=False)
        #plt.savefig('summary_plot.jpg')
        #shap.plots.heatmap(shap_values, show=False)
        #plt.savefig('heatmap_plot.jpg')

        return { 
            'preds_info': res,
            'feature_importances': feature_importances
        }
        print('Done!')

    def __addfeatures__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function for adding features to given data"""

        df = df.copy()
        df['sum_handlers'] = df['n_handlers_7'] + df['n_handlers_15'] + df['n_handlers_30'] + df['n_handlers_30']
        df['sum_days_0_8'] = df['0_1'] + df['1_2'] + df['2_3'] + df['3_4'] + df['4_5'] + df['5_6'] + df['6_7'] + df['7_8']
        df['sum_change_date'] = df['change_date_15'] + df['change_date_30'] + df['change_date_7']
        df['mon3-mon2'] = df['mon3'] - df['mon2']
        df['mon3-mon1'] = df['mon3'] - df['mon1']
        df['mon2-mon1'] = df['mon2'] - df['mon1']
        df['sum_agreements'] = df['agreement_1'] + df['agreement_2'] + df['agreement_3']

        return df

    def __readset__(self, csv_path: str) -> pd.DataFrame:
        """Function for reading and preparing data by passing path to csv file"""

        df = pd.read_csv(csv_path)

        # ---------------------------------------------------------------------------------------
        df = df.drop(columns=['Количество позиций'], errors='ignore')

        rename_list = ['provider', 'material', 'category_manager', 'operation_manager', 'factory', 'purchase_org', 'purchase_group',
               'balance_unit', 'unit', 'material_group', 'supply_variant', 'NRP', 'duration', 'before_supply', 'mon1', 'mon2', 'mon3',
               'weekday2', 'sum', 'quantity', 'n_handlers_7', 'n_handlers_15', 'n_handlers_30', 'agreement_1', 'agreement_2', 'agreement_3',
               'change_date_7', 'change_date_15', 'change_date_30', 'cancel_deblock', 'change_paper', 'change', 'n_loops_agreement', 'n_changes',
               '0_1', '1_2', '2_3', '3_4', '4_5', '5_6', '6_7', '7_8']

        rename_dict = {df.columns[i]: rename_list[i] for i in range(len(df.columns))}
        df = df.rename(columns=rename_dict)
        # --------------------------------------------------------------------------------------

        df = self.__addfeatures__(df)

        return df
    