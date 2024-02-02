from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_aic_bic(model, X, y):
    predictions = model.predict(X)
    rss = mean_squared_error(y, predictions) * len(y)  # Residual Sum of Squares
    n = len(y)  # number of observations
    k = len(model.coef_) + 1  # number of parameters (coefficients + intercept)

    # Calculate Log-Likelihood
    ll = -n/2 * (np.log(2*np.pi) + np.log(rss/n) + 1)

    # Calculate AIC
    aic = 2*k - 2*ll

    # Calculate BIC
    bic = np.log(n)*k - 2*ll

    return aic, bic