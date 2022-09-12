import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(X, y):
    # number of observations/points
    n = np.size(X)

    # mean of X and y vector
    X_mean = np.mean(X)
    y_mean = np.mean(y)

    # Calculate the terms needed for numerator and denominator of beta1
    Xy_cov = (X - X_mean) * (y - y_mean)
    x_var = (X - X_mean) ** 2

    # calculating regression coefficients
    beta1 = Xy_cov.sum() / x_var.sum()
    beta0 = y_mean - (beta1 * X_mean)

    return beta0, beta1


def plot_regression_line(X, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(X, y, color="m", marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * X

    # plotting the regression line
    plt.plot(X, y_pred, color="g")

    # putting labels
    plt.xlabel('No. of Advertisement')
    plt.ylabel('Sales')
    plt.title('No. of Advertisements on TV Vs Sales')

    # function to show plot
    plt.show()


def main():
    # Import and display first five rows of advertising dataset
    advertise = pd.read_csv('dataset/Advertising.csv')
    print(advertise.head())

    # create X (features) and y (response)
    X = advertise['TV']
    y = advertise['Sales']

    # estimating coefficients
    r_coeff = estimate_coeff(X, y)
    print("Estimated coefficients:\nbeta_0 = {}  \
              \nbeta_1 = {}".format(r_coeff[0], r_coeff[1]))

    # plotting regression line
    plot_regression_line(X, y, r_coeff)

if __name__ == "__main__":
    main()
