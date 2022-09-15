import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import minmax

class Predict:
    """ 
    Predict car price based on mileage
    Get theta thanks to the train class
    We use the following hypothesis to predict the price :
        estimateP rice(mileage) = θ0 + (θ1 * mileage)
    """

    def __init__(self, df: pd.DataFrame, theta: list):
        self.df = df
        self.x = np.array(self.df['km'], dtype='float64').reshape(-1, 1)
        self.x_norm = minmax(self.x)
        self.y = np.array(self.df['price'], dtype='float64').reshape(-1, 1)
        self.theta = np.array([theta[0], theta[1]]).reshape(-1, 1)
        
    def predict_price(self, mileage: int):
        """ estimate Price(mileage) = θ0 + (θ1 ∗ mileage) """
        pred = self.theta[0] + self.theta[1] * minmax(np.array([mileage]).reshape(-1, 1), np.min(self.x), np.max(self.x))
        return int(pred)

    def plot_regression(self):
        """ predicted response vector
        estimate Price(mileage) = θ0 + (θ1 ∗ mileage)
        """
        y_pred = (self.theta[1] * np.array(self.x_norm)) + self.theta[0]
        plt.plot(self.x, y_pred, color= "limegreen")

    def plot_estimated_price(self, mileage: int, price: float, xlim: int, ylim: int):
        plt.stem([mileage], [price], bottom=ylim, orientation='vertical', linefmt='c--', markerfmt="c:D")
        plt.stem([price], [mileage], bottom=xlim, orientation='horizontal', linefmt='c--', markerfmt="c:D")

    def plot(self, mileage: int, price: int):
        plt.plot(self.x, self.y, 'o')
        self.plot_regression()
        # we calculate the limit (on axis x and y) of the intersection lines of the prediction
        # for a better scale of visualization (default limit would be 0 or less)
        xlim = min(self.x) - (min(self.x) * 15 / 100)
        ylim = min(self.y) - (min(self.y) * 15 / 100)
        plt.xlim(int(xlim))
        plt.ylim(int(ylim))
        self.plot_estimated_price(mileage, price, xlim, ylim)
        plt.title(f"Theta = {self.theta[0]} {self.theta[1]}", fontdict={'fontsize': 9})
        plt.suptitle(f"Car price based on mileage : {price}$ for {mileage}km")
        plt.xlabel('mileage')
        plt.ylabel('price')
        plt.show()
