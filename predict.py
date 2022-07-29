import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Predict:
    """ 
    Predict car price based on mileage
    Get theta thanks to the train class
    We use the following hypothesis to predict the price :
        estimateP rice(mileage) = θ0 + (θ1 * mileage)
    """
    def __init__(self, df: pd.DataFrame, theta: list):
        self.df = df
        self.x = np.array(self.df['km'])
        self.x_norm = [self.normalize_minmax(element, self.x) for element in self.x]
        self.y = np.array(self.df['price'])
        self.theta = [theta[0], theta[1]]
        
    def plot_regression(self):
        """ predicted response vector
        estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)
        f(x) = ax + b
        """
        y_pred = (self.theta[1] * np.array(self.x_norm, float)) + self.theta[0]
        plt.plot(self.x, y_pred, color= "g")

    def plot_estimated_price(self, mileage, price, xlim, ylim):
        plt.stem([mileage], [price], bottom=ylim, orientation='vertical', linefmt='c--', markerfmt="c:D")
        plt.stem([price], [mileage], bottom=xlim, orientation='horizontal', linefmt='c--', markerfmt="c:D")

    def plot(self, mileage, price):
        plt.plot(self.x, self.y, 'o')
        xlim = min(self.x) - (min(self.x) * 15 / 100)
        ylim = min(self.y) - (min(self.y) * 10 / 100)
        plt.xlim(xlim)
        plt.ylim(ylim)
        self.plot_regression()
        self.plot_estimated_price(mileage, price, xlim, ylim)
        plt.xlabel('km')
        plt.ylabel('price')
        plt.suptitle(f"Car price based on mileage : {price}$ for {mileage}km")
        plt.title(f"Theta = {self.theta}", fontdict={'fontsize': 9})
        plt.show()

    def normalize_minmax(self, value, lst):
        return (value - min(lst)) / (max(lst) - min(lst))

    def predict_price(self, mileage):
        """ f(x) = ax + b """
        return int(self.theta[1] * self.normalize_minmax(mileage, self.x) + self.theta[0])
