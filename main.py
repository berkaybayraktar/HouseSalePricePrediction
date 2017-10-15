import pandas as pd
import numpy as np
import operator
import csv
import random
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split

df_train = pd.read_csv('train.csv')

#Deleting outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

categorical_columns = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'MoSold', 'YrSold']

numerical_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch','ScreenPorch', 'PoolArea', 'MiscVal']

all_columns = categorical_columns + numerical_columns

# print(df_train['Neighborhood'].sort_values().unique())

# Missing rows
print("Missing Data")
df_train_na = (df_train.isnull().sum() / len(df_train)) * 100
df_train_na = df_train_na.drop(df_train_na[df_train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': df_train_na})
print(missing_data.head(20))

# f, ax = plt.subplots()
# plt.xticks(rotation='90')
# sns.barplot(x=df_train_na.index, y=df_train_na)
# plt.xlabel('Features')
# plt.ylabel('Percent of missing values')
# plt.title('Percent missing data by feature')
# plt.show()

# Fill missing values
df_train["PoolQC"] = df_train["PoolQC"].fillna("None")
df_train["MiscFeature"] = df_train["MiscFeature"].fillna("None")
df_train["Alley"] = df_train["Alley"].fillna("None")
df_train["Fence"] = df_train["Fence"].fillna("None")
df_train["FireplaceQu"] = df_train["FireplaceQu"].fillna("None")
df_train["LotFrontage"] = df_train["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_train[col] = df_train[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_train[col] = df_train[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_train[col] = df_train[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_train[col] = df_train[col].fillna('None')
df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)

# Normalize categorical columns
# Categorical olan alanları Numeric e çevir
for column_name in categorical_columns:
    lbl = LabelEncoder()
    lbl.fit(list(df_train[column_name].values))
    df_train[column_name] = lbl.transform(list(df_train[column_name].values))

train_descriptions = {}
for column_name in all_columns:
    train_descriptions[column_name] = {'mean': df_train[column_name].mean(), 'std': df_train[column_name].std(), 'median': df_train[column_name].median()}

# Normalize numerical columns
for column_name in all_columns:
    std = train_descriptions[column_name]['std']
    mean = train_descriptions[column_name]['mean']
    print(column_name, 'mean:', mean, 'std:', std)
    for index in range(df_train[column_name].values.size):
        df_train[column_name].values[index] = (df_train[column_name].values[index] - mean) / std


df_test = pd.read_csv('test.csv')

df_test["PoolQC"] = df_test["PoolQC"].fillna("None")
df_test["MiscFeature"] = df_test["MiscFeature"].fillna("None")
df_test["Alley"] = df_test["Alley"].fillna("None")
df_test["Fence"] = df_test["Fence"].fillna("None")
df_test["FireplaceQu"] = df_test["FireplaceQu"].fillna("None")
df_test["LotFrontage"] = df_test["LotFrontage"].fillna(train_descriptions['LotFrontage']['median'])
# df_test["LotFrontage"] = df_test["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    df_test[col] = df_test[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    df_test[col] = df_test[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    df_test[col] = df_test[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    df_test[col] = df_test[col].fillna('None')
df_test["MasVnrType"] = df_test["MasVnrType"].fillna("None")
df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)

for column_name in categorical_columns:
    lbl = LabelEncoder()
    lbl.fit(list(df_test[column_name].values))
    df_test[column_name] = lbl.transform(list(df_test[column_name].values))

for column_name in all_columns:
    df_test[column_name] = df_test[column_name].astype(float)
    std = train_descriptions[column_name]['std']
    mean = train_descriptions[column_name]['mean']
    print(column_name, 'mean:', mean, 'std:', std)
    for index, value in df_test[column_name].iteritems():
        df_test[column_name].values[index] = (value - mean) / std



# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(df_train[all_columns], df_train["SalePrice"])

predictions = knn.predict(df_test[all_columns])

for index, row in df_test.iterrows():
    print(row["Id"],",",predictions[index])

result = pd.DataFrame()
result['Id'] = df_test['Id']
result['SalePrice'] = predictions
result.to_csv('result.csv', index=False)


# k = 0 en yakın olan
# for index, row in df_test.iterrows():
#
#     best_match_similarity = 0
#     best_match_similarity_id = 0
#
#     for train_index, train_row in normalised_train.iterrows():
#         similarity = 0
#         for column_name in all_columns:
#             similarity += (row[column_name] - train_row[column_name]) ** 2
#
#         similarity = np.sqrt(similarity)
#
#         if similarity > best_match_similarity:
#             best_match_similarity = similarity
#             best_match_similarity_id = train_row["SalePrice"]
#
#     print("testId: ", row["Id"], "testPrice: ", row["SalePrice"], "estimatedPrice: ", best_match_similarity_id,
#           "similarity: ", best_match_similarity)




# Distribution Plot
# sns.distplot(df_train.LotFrontage[~df_train.LotFrontage.isnull()], fit=norm)
# (mu, sigma) = norm.fit(df_train.LotFrontage[~df_train.LotFrontage.isnull()])
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(df_train.LotFrontage.mean(),
#                                                                         df_train.LotFrontage.std())], loc='best')
# plt.ylabel('Frequency')
# plt.title('LotFrontage distribution')
# fig = plt.figure()
# res = stats.probplot(df_train.LotFrontage[~df_train.LotFrontage.isnull()], plot=plt)
# plt.show()


# Remove Id column
# train.drop("Id", axis=1, inplace=True)
# print(df_train['MSSubClass'].describe())
#
# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data)
#
