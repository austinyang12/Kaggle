import pandas as pd
import seaborn as sns
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.optimizers import RMSprop

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

y = df_train['RainToday']

df_train = df_train.drop(['RainToday'], axis=1)
#df_train = df_train.drop(['MinTemp'], axis=1)
#df_train = df_train.drop(['WindDir9am'], axis=1)
#df_train = df_train.drop(['WindDir3pm'], axis=1)

categorical = [var for var in df_train.columns if df_train[var].dtype == object]

df_train['Date'] = pd.to_datetime(df_train['Date'])
df_test['Date'] = pd.to_datetime(df_test['Date'])

df_train['Year'] = df_train['Date'].dt.year
df_train['Month'] = df_train['Date'].dt.year
#df_train['Day'] = df_train['Date'].dt.year

df_test['Year'] = df_test['Date'].dt.year
df_test['Month'] = df_test['Date'].dt.year
#df_test['Day'] = df_test['Date'].dt.year

df_train.drop('Date', axis=1, inplace=True)
df_test.drop('Date', axis=1, inplace=True)

pd.get_dummies(df_train.Location, drop_first=True)
pd.get_dummies(df_train.WindGustDir, drop_first=True, dummy_na=True)
pd.get_dummies(df_train.WindDir9am, drop_first=True, dummy_na=True)
pd.get_dummies(df_train.WindDir3pm, drop_first=True, dummy_na=True)

pd.get_dummies(df_test.Location, drop_first=True)
pd.get_dummies(df_test.WindGustDir, drop_first=True, dummy_na=True)
pd.get_dummies(df_test.WindDir9am, drop_first=True, dummy_na=True)
pd.get_dummies(df_test.WindDir3pm, drop_first=True, dummy_na=True)

numerical = [var for var in df_train.columns if df_train[var].dtype != object]


IQR = df_train.Evaporation.quantile(0.75) - df_train.Evaporation.quantile(0.25)
Lower_fence_1 = df_train.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence_1 = df_train.Evaporation.quantile(0.75) + (IQR * 3)
#print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_1, upperboundary=Upper_fence_1))

IQR = df_train.WindGustSpeed.quantile(0.75) - df_train.WindGustSpeed.quantile(0.25)
Lower_fence_2 = df_train.WindGustSpeed.quantile(0.25) - (IQR * 3)
Upper_fence_2 = df_train.WindGustSpeed.quantile(0.75) + (IQR * 3)
#print('WindGustSpeed outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_2, upperboundary=Upper_fence_2))

IQR = df_train.WindSpeed9am.quantile(0.75) - df_train.WindSpeed9am.quantile(0.25)
Lower_fence_3 = df_train.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence_3 = df_train.WindSpeed9am.quantile(0.75) + (IQR * 3)
#print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_3, upperboundary=Upper_fence_3))


IQR = df_train.WindSpeed3pm.quantile(0.75) - df_train.WindSpeed3pm.quantile(0.25)
Lower_fence_4 = df_train.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence_4 = df_train.WindSpeed3pm.quantile(0.75) + (IQR * 3)
#print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_4, upperboundary=Upper_fence_4))


IQR = df_test.Evaporation.quantile(0.75) - df_test.Evaporation.quantile(0.25)
Lower_fence_5 = df_test.Evaporation.quantile(0.25) - (IQR * 3)
Upper_fence_5 = df_test.Evaporation.quantile(0.75) + (IQR * 3)
#print('Evaporation outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_5, upperboundary=Upper_fence_5))

IQR = df_test.WindGustSpeed.quantile(0.75) - df_test.WindGustSpeed.quantile(0.25)
Lower_fence_6 = df_test.WindGustSpeed.quantile(0.25) - (IQR * 3)
Upper_fence_6 = df_test.WindGustSpeed.quantile(0.75) + (IQR * 3)
#print('WindGustSpeed outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_6, upperboundary=Upper_fence_6))

IQR = df_test.WindSpeed9am.quantile(0.75) - df_test.WindSpeed9am.quantile(0.25)
Lower_fence_7 = df_test.WindSpeed9am.quantile(0.25) - (IQR * 3)
Upper_fence_7 = df_test.WindSpeed9am.quantile(0.75) + (IQR * 3)
#print('WindSpeed9am outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_7, upperboundary=Upper_fence_7))


IQR = df_train.WindSpeed3pm.quantile(0.75) - df_train.WindSpeed3pm.quantile(0.25)
Lower_fence_8 = df_test.WindSpeed3pm.quantile(0.25) - (IQR * 3)
Upper_fence_8 = df_test.WindSpeed3pm.quantile(0.75) + (IQR * 3)
#print('WindSpeed3pm outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence_8, upperboundary=Upper_fence_8))


X = df_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

numerical = [col for col in X_train.columns if X_train[col].dtypes != object]


for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

for df1 in [df_test]:
    for col in numerical:
        col_median = X_test[col].median()
        df1[col].fillna(col_median, inplace=True)

categorical.pop(0) # pop 'Date'

for df2 in [X_train, X_test]:
    df2['Location'].fillna(X_train['Location'].mode()[0], inplace=True)
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)

for df2 in [df_test]:
    df2['Location'].fillna(df_test['Location'].mode()[0], inplace=True)
    df2['WindGustDir'].fillna(df_test['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(df_test['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(df_test['WindDir3pm'].mode()[0], inplace=True)

def max_value(df3, variable, top):
    return np.where(df3[variable] > top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Evaporation'] = max_value(df3, 'Evaporation', 9.5798)
    df3['WindGustSpeed'] = max_value(df3, 'WindGustSpeed', 91)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

for df3 in [df_test]:
    df3['Evaporation'] = max_value(df3, 'Evaporation', 9.5798)
    df3['WindGustSpeed'] = max_value(df3, 'WindGustSpeed', 91)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)


X_train = pd.concat([X_train[numerical],
                    pd.get_dummies(X_train.Location),
                    pd.get_dummies(X_train.WindGustDir),
                    pd.get_dummies(X_train.WindDir9am),
                    pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical],
                    pd.get_dummies(X_test.Location),
                    pd.get_dummies(X_test.WindGustDir),
                    pd.get_dummies(X_test.WindDir9am),
                    pd.get_dummies(X_test.WindDir3pm)], axis=1)

df_test = pd.concat([df_test[numerical],
                    pd.get_dummies(df_test.Location),
                    pd.get_dummies(df_test.WindGustDir),
                    pd.get_dummies(df_test.WindDir9am),
                    pd.get_dummies(df_test.WindDir3pm)], axis=1)

cols = X_train.columns

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df_test = scaler.fit_transform(df_test)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

df_test = pd.DataFrame(df_test, columns=[cols])

'''
selector = SelectKBest(chi2, k=3)
print(X_train)
print("")
print(y_train)
selector.fit(X_train, y_train)
X_new = selector.transform(X_train)
print(X_train.columns[selector.get_support(indices=True)])
print('HERE')

UnivariateFeatureSelection = SelectKBest(chi2, k=5).fit(X_train,y_train)

diccionario = {key:value for (key, value) in zip(UnivariateFeatureSelection.scores_, X_train.columns)}
print(sorted(diccionario.items(), reverse=True))
'''

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=3, n_jobs=2)
forest.fit(X_train, y_train)

y_pred_test = forest.predict(X_test)


print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

ans = forest.predict(df_test)

#logreg100 = LogisticRegression(C=100, solver='liblinear', random_state=0)

'''
model = Sequential([
    Dense(128, input_dim=115),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(1),
    Activation('tanh')
])

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=rmsprop,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


model.fit(X_train, y_train, epochs=10, batch_size=256)

loss, accuracy = model.evaluate(X_test, y_test)

ans = model.predict(df_test)

print(ans)

for i in range(len(ans)):
    if ans[i] < 0.5:
        ans[i] = 0
    else:
        ans[i] = 1

print(ans)
'''

#logreg100.fit(X_train, y_train)

#y_pred_test = logreg100.predict(X_test)
#ans = logreg100.predict(df_test)

#print(logreg100.predict_proba(X_test)[:, 0])
#print(logreg100.predict_proba(X_test)[:, -1])

#print('Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

#print(X_train.describe())

#y_pred_train = logreg100.predict(X_train)

#print(y_pred_train)

#print('Training-set accuracy score: {0:0.4f}'.format(accuracy_score(y_train, y_pred_train)))



ans = pd.DataFrame(ans.astype(int), columns = ['RainToday'])
ans.to_csv('myAns.csv', index_label = 'ID')

print('done')

