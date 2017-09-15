import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as plt

df = pd.read_csv('http://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv', index_col=0)
print(df.head())

# The X is the predictor variable and the Y is the response
# therefore, X is the FACT-G score and Y is the EQ-5D health-related Quality of life index
# in this instance, the Y is just a possible range from 0 to 1.

# the .Employed refers to a column header 'Employed' in the csv file
y = df.Employed  # Response
x = df.GNP  # Predictor
x = sm.add_constant(x)  # Adds constant term to the predictor
print(x.head())

est = sm.OLS(y,x)
est = est.fit()
est.summary()

print(est.params)

# %pylab inline

# Pick 100 points equally spaced from the min to the max
X_prime = np.linspace(x.GNP.min(), X.GNP.max(), 100)[:, np.newaxis]
X_prime = sm.add_constant(X_prime)  # Add a constant as we did before

# Calculate the predicted values
y_hat = est.predict(X_prime)

plt.scatter(X.GNP, y, alpha=0.3)   # Plot the raw data
plt.xlabel("Gross National Product")
plt.ylabel("Total Employment")
plt.plot(X_prime[:, 1], y_hat, 'r', alpha=0.9)  # adds the regression line

plt.show()

# est = smf.ols(formula='Employed ~ GBP', data=df).fit()
# print(est.summary())
