import pandas as pd
import numpy as np

import xgboost as xgb 

from sklearn import preprocessing
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import calibration
import warnings

warnings.filterwarnings("ignore")

grid = False

df_train = pd.read_csv("projeto4_telecom_treino.csv")
df_test = pd.read_csv("projeto4_telecom_teste.csv")

df_train.drop(["Unnamed: 0"], axis=1, inplace=True)
df_test.drop(["Unnamed: 0"], axis=1, inplace=True)

df_train = pd.get_dummies(df_train, prefix=["area_code", "state"], columns=["area_code", "state"])
df_test = pd.get_dummies(df_test, prefix=["area_code", "state"], columns=["area_code", "state"])

df_train.loc[df_train["churn"] == "no", "churn"] = 0
df_train.loc[df_train["churn"] == "yes", "churn"] = 1

df_train.loc[df_train["international_plan"] == "no", "international_plan"] = 0
df_train.loc[df_train["international_plan"] == "yes", "international_plan"] = 1

df_train.loc[df_train["voice_mail_plan"] == "no", "voice_mail_plan"] = 0
df_train.loc[df_train["voice_mail_plan"] == "yes", "voice_mail_plan"] = 1


df_test.loc[df_test["churn"] == "no", "churn"] = 0
df_test.loc[df_test["churn"] == "yes", "churn"] = 1

df_test.loc[df_test["international_plan"] == "no", "international_plan"] = 0
df_test.loc[df_test["international_plan"] == "yes", "international_plan"] = 1

df_test.loc[df_test["voice_mail_plan"] == "no", "voice_mail_plan"] = 0
df_test.loc[df_test["voice_mail_plan"] == "yes", "voice_mail_plan"] = 1


y_train = df_train["churn"].values
x_train = df_train.drop(["churn"], axis=1)

y_test = df_test["churn"].values
x_test = df_test.drop(["churn"], axis=1)

x_train_norm = preprocessing.normalize(x_train, axis=0, norm="l1")
x_test_norm = preprocessing.normalize(x_test, axis=0, norm="l1")

pca = decomposition.PCA(3)
x_train_pca = pca.fit_transform(x_train_norm)
x_test_pca = pca.transform(x_test_norm)

x_train["pca_0"] = x_train_pca[:,0]
x_train["pca_1"] = x_train_pca[:,1]
x_train["pca_2"] = x_train_pca[:,2]

x_test["pca_0"] = x_test_pca[:,0]
x_test["pca_1"] = x_test_pca[:,1]
x_test["pca_2"] = x_test_pca[:,2]

model_extra_tree = ensemble.ExtraTreesClassifier(class_weight="balanced_subsample", bootstrap=True, random_state=42)
model_extra_tree_fitted = model_extra_tree.fit(x_train, y_train)

model_select_feat = feature_selection.SelectFromModel(model_extra_tree_fitted, prefit=True)

x_train = model_select_feat.transform(x_train)
x_test = model_select_feat.transform(x_test)

model_x_gradient_boosting = xgb.XGBClassifier(missing=np.nan, max_depth=6, n_estimators=350, learning_rate=0.025, nthread=8, subsample=0.95, colsample_bytree=0.85, seed=42)

model_x_gradient_boosting.fit(x_train, y_train)

result = model_x_gradient_boosting.score(x_test, y_test)

print(result)

