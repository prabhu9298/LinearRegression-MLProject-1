import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV data
df=pd.read_csv(r"C:\Users\PRABHU DAS\Downloads\SOCR-HeightWeight.csv")
df.head()
# Visualize the data
plt.scatter(df["Weight(Pounds)"],df["Height(Inches)"])
plt.ylabel("Height(Inches)")
plt.xlabel("Weight(Pounds)")
# Check correlation
df.corr()
#seaborn for visualisation
import seaborn as sns
sns.pairplot(df)
# Split into independent and dependent features
x=df[["Weight(Pounds)"]]#independent feature
y=df["Height(Inches)"]#dependent feature
# Split data into training and testing sets
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) #to prevent data leakage
#Apply linear regression
from sklearn.linear_model import LinearRegression
regression=LinearRegression(n_jobs=-1)
regression.fit(x_train,y_train)
# Print coefficients and intercept
print("coefficient or slope :",regression.coef_)
print("intercept:",regression.intercept_)
# Plot best fit line
plt.scatter(x_train,y_train)
plt.plot(x_train,regression.predict(x_train))
# Make predictions on test data
y_pred=regression.predict(x_test)
# Evaluate model performance
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse) #root mean squared error
print("Mean square error:",mse)
print("Mean absolute error:",mae)
print("Root  Mean Square Error:",rmse)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
# Predict for a new value
print(regression.predict(scaler.transform([[124.8742]])))


