import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as plt


df = pd.read_csv('C:\\Users\\mrwilliams\\Documents\\Program_Projects\\Ordinary_Least_Squares\\FACT-G.csv', index_col=0)

# Rescale FACT-G scores to 0 - 100 scale as described in Model 1 on page 3 of Teckle et al.
norm_df = df.copy()
scaled_score = []
max_value = 108
min_value = 0
for Score in norm_df.Score:
    scaled_score.append((Score - min_value) / (max_value - min_value) * 100)

norm_df['Scaled_Score'] = scaled_score
print(norm_df.head())

# The X is the predictor variable and the Y is the response
# therefore, X is the FACT-G score and Y is the EQ-5D health-related Quality of life index
# in this instance, the Y is just a possible range from 0 to 1.


# y = df.HRQoL_Index  # Response
# y = np.linspace(0, 1, 100)
# x = df.Score  # Predictor
# x = sm.add_constant(x)  # Adds constant term to the predictor
# print(x.head())

# est = sm.OLS(y,x)
# est = est.fit()
# print(est.summary())

# print(est.params)

# # %pylab inline

# # Pick 100 points equally spaced from the min to the max
# X_prime = np.linspace(x.GNP.min(), X.GNP.max(), 100)[:, np.newaxis]
# X_prime = sm.add_constant(X_prime)  # Add a constant as we did before

# # Calculate the predicted values
# y_hat = est.predict(X_prime)

# plt.scatter(X.GNP, y, alpha=0.3)   # Plot the raw data
# plt.xlabel("Gross National Product")
# plt.ylabel("Total Employment")
# plt.plot(X_prime[:, 1], y_hat, 'r', alpha=0.9)  # adds the regression line

# plt.show()

# # est = smf.ols(formula='Employed ~ GBP', data=df).fit()
# print(est.summary())
