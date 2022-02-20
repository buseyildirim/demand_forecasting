
#####################################################
# Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge
# https://www.kaggle.com/c/demand-forecasting-kernels-only

# Farklı store için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)

usd_kur = pd.read_csv('datasets/demand_forecasting/usd.csv', parse_dates=['date'])
usd_kur.info()
usd_kur = usd_kur[["date", "dolar_kuru"]]
df = df.merge(usd_kur, how="inner")
train = train.merge(usd_kur, how="inner")
test = test.merge(usd_kur, how="inner")

#####################################################
# EDA
#####################################################
df.info()
df["dolar_kuru"] = df["dolar_kuru"].str.replace(',', '.')

df["dolar_kuru"] = df["dolar_kuru"].astype("float64")

check_df(train)

check_df(test)

check_df(df)

df["date"].min(), df["date"].max()

df["dolar_kuru"].min(), df["dolar_kuru"].max()

# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


#####################################################
# FEATURE ENGINEERING
#####################################################


########################
# Date Features
########################


def create_date_features(df,date_column):
    df['month'] = df[date_column].dt.month
    df['day_of_month'] = df[date_column].dt.day
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.weekofyear
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['year'] = df[date_column].dt.year
    df["is_wknd"] = df[date_column].dt.weekday // 4
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['quarter'] = df[date_column].dt.quarter
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    return df

df = create_date_features(df,"date")

check_df(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag Features
########################

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
check_df(df)


########################
# Rolling Mean Features
########################

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546])
df.tail()


########################
# Exponentially Weighted Mean Features
########################


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean()) \
                + random_noise(dataframe)
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

check_df(df)

########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

########################
# Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values)
check_df(df)


#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


train = df.loc[(df["date"] < "2017-01-01"), :]

val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

######################################################
# Base Models
######################################################

import lightgbm as lgbm

lgbm_model = lgbm.LGBMRegressor().fit(X_train, Y_train)
y_pred = lgbm_model.predict(X_val)

# Validation rmse
np.sqrt(mean_squared_error(np.expm1(y_pred), np.expm1(Y_val)))

smape(np.expm1(y_pred), np.expm1(Y_val))
######################################################
# Optimizasyon
######################################################

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "max_depth": [None],
                   "n_estimators": [500, 1000],
                   "colsample_bytree": [0.8, 1]}

best_model = GridSearchCV(lgbm_model, lightgbm_params, cv=5, n_jobs=-1, verbose=False).fit(X_train, Y_train)

best_model.best_params_

final_model = lgbm_model.set_params(**best_model.best_params_)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, Y_train, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE (After): {round(rmse, 4)}")

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.01,
              'feature_fraction': 1,
              'max_depth': None,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Değişken önem düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

########################
# Final Model
########################


train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.01,
              'feature_fraction': 1,
              'max_depth': None,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = model.predict(X_test, num_iteration=model.best_iteration)

submission_df = test.loc[:, ['id', 'sales']]
submission_df['sales'] = np.expm1(test_preds)
submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv('submission_demand.csv', index=False)
submission_df.head(20)
