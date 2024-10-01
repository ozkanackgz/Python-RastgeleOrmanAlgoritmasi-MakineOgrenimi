import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("veri.csv", encoding="ISO-8859-9")





print(data.head())



X = data[['yil', 'internetkullanicis', 'emailkullanicis', 'sosyalmkullanicis']]
y = data['ıncidents']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



train_results = pd.DataFrame({'Gerçek Değerler': y_train, 'Tahmin Edilen Değerler': model.predict(X_train)})
print("Eğitim Verileri ile Tahmin Edilen Değerler:")
print(train_results)




print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')