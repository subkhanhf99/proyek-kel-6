import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pickle

pd.__version__
# Load the csv file
df = pd.read_csv("poverty.csv")

print(df.head())

# Select independent and dependent variable
X = df[['Cases','Deaths','Population','W_Male','W_Female','B_Male','B_Female','H_Male','H_Female','I_Male','I_Female','A_Male','A_Female','NH_Male','NH_Female']].values
y = df['Poverty'].values

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Feature scaling
scaler = StandardScaler().fit(X_train)
rescaled_X_train = scaler.transform(X_train)
model = GradientBoostingRegressor(n_estimators=400)
A = model.fit(rescaled_X_train, y_train)


# Fit the model


# Make pickle file of our model
pickle.dump(A, open("model.pkl", "wb"))