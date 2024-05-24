# Heterogeneous Ensemble Methods

This repository contains a Jupyter Notebook that explores heterogeneous ensemble methods in machine learning. By combining different types of models, we can leverage their unique strengths and improve predictive performance. The notebook demonstrates the process of training individual models and then combining them using voting and stacking ensembles, showcasing their effectiveness with practical examples.

## Overview

### Heterogeneous Ensembles
Heterogeneous ensembles use a variety of model types to create a more robust predictive model. This notebook covers:
- **Linear Regression**
- **Decision Tree Regressor**
- **Support Vector Regressor (SVR)**
- **Voting Ensemble**
- **Stacking Ensemble**

## Notebook Content

### 1. Training the Individual Models

#### Import Libraries and Data
First, we import necessary libraries and load the dataset. We visualize the data to understand its distribution.

```python
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://github.com/Explore-AI/Public-Data/blob/master/house_price_by_area.csv?raw=true")
X = df["LotArea"] # Independent variable
y = df["SalePrice"] # Dependent variable

plt.scatter(X, y)
plt.title("House Price vs Area")
plt.xlabel("Lot Area in m$^2$")
plt.ylabel("Sale Price in Rands")
plt.show()
```

#### Pre-processing
We normalize the data to ensure that the models perform optimally.

```python
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(np.array(X)[:, np.newaxis])
y_scaled = y_scaler.fit_transform(np.array(y)[:, np.newaxis])

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=6)
```

#### a) Linear Regression
We train a linear regression model and evaluate its performance.

```python
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

y_pred = lin_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(lin_reg.predict(x_domain))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Linear Regression")
plt.legend()
plt.show()
```

#### b) Decision Tree Regressor
We train a decision tree regressor and evaluate its performance.

```python
regr_tree = DecisionTreeRegressor(max_depth=3)
regr_tree.fit(x_train, y_train)

y_pred = regr_tree.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(regr_tree.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Decision Tree")
plt.legend()
plt.show()
```

#### c) Support Vector Regressor
We train a support vector regressor and evaluate its performance.

```python
sv_reg = SVR(kernel='rbf', gamma='auto')
sv_reg.fit(x_train, y_train[:, 0])

y_pred = sv_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(sv_reg.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Support Vector Regression")
plt.legend()
plt.show()
```

### 2. Heterogeneous Ensembling in Python

#### a) Voting Ensemble
We build a voting ensemble that combines the outputs of the linear regression, decision tree, and support vector regressor models.

```python
from sklearn.ensemble import VotingRegressor

models = [("LR", lin_reg), ("DT", regr_tree), ("SVR", sv_reg)]
model_weightings = np.array([0.1, 0.3, 0.6])

v_reg = VotingRegressor(estimators=models, weights=model_weightings)
v_reg.fit(x_train, y_train[:, 0])

y_pred = v_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(v_reg.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Voting Ensemble Regression")
plt.legend()
plt.show()
```

#### b) Stacking Ensemble
We build a stacking ensemble that uses a meta-learner to combine the outputs of the individual models.

```python
from sklearn.ensemble import StackingRegressor

models = [("LR", lin_reg), ("DT", regr_tree), ("SVR", sv_reg)]
meta_learner_reg = LinearRegression()

s_reg = StackingRegressor(estimators=models, final_estimator=meta_learner_reg)
s_reg.fit(x_train, y_train[:, 0])

y_pred = s_reg.predict(x_test)
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

# Plotting
x_domain = np.linspace(min(x_train), max(x_train), 100)
y_pred_rescaled = y_scaler.inverse_transform(s_reg.predict(x_domain).reshape(-1, 1))
x_rescaled = x_scaler.inverse_transform(x_domain)

plt.figure()
plt.scatter(X, y)
plt.plot(x_rescaled, y_pred_rescaled, color="red", label='predictions')
plt.xlabel("LotArea in m$^2$")
plt.ylabel("SalePrice in Rands")
plt.title("Stacking Ensemble Regression")
plt.legend()
plt.show()
```

### Results
The notebook demonstrates how heterogeneous ensemble methods can improve predictive performance by combining the strengths of different models. The stacking ensemble, in particular, achieves the best RMSE, illustrating the power of this approach.

## Usage
To run this notebook, clone this repository and open the notebook in Jupyter:

```bash
git clone https://github.com/yourusername/Heterogeneous-Ensemble-Methods.git
cd Heterogeneous-Ensemble-Methods
jupyter notebook
```

## Conclusion
This notebook provides a comprehensive introduction to heterogeneous ensemble methods, showing how to build, train, and evaluate various ensemble models. It serves as a valuable resource for data scientists and machine learning practitioners interested in enhancing their models' predictive performance.

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to improve this repository.
