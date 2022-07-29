from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

NB_FRAME = 10

class Train:
    """
    Read dataset and train the model thanks to gradient descent algorithm
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x = list(map(float, self.df['km']))
        self.x_norm = [self.normalize_minmax(element, self.x) for element in self.x]
        self.y = list(map(float, self.df['price']))
        self.theta = [0, 0]
        self.theta_history = []
        self.losses = []

    
    def normalize_minmax(self, value, lst):
        """ Normalize data in 0-1 interval
        Not normalized data makes the minimum point really difficult to reach
        """
        return (value - min(lst)) / (max(lst) - min(lst))
    
    def predict_price(self, mileage):
        """ f(x) = ax + b """
        return (self.theta[1] * mileage + self.theta[0])
   
    def calculate_theta_mean(self, m: int, theta_index: int):
        """ mean of errors :
            - theta[0]: mean of -> differences beetween estimated price and real price
            - theta[1]: mean of -> differences beetween estimated price and real price multiplied by mileage
        """
        result = 0
        mileage = self.x_norm
        price = self.y
        if theta_index == 0:
            for i in range(m):
                result += (self.predict_price(mileage[i]) - price[i])
        elif theta_index == 1:
            for i in range(m):
                result += ((self.predict_price(mileage[i]) - price[i]) * mileage[i])
        return result / m

    def calculate_loss(self):
        m = len(self.y)
        sum_losses = 0
        for i in range(m):
            sum_losses += ((self.predict_price(self.x_norm[i]) - self.y[i])) ** 2 / m
        return sum_losses

    def gradient_descent(self, learning_rate: float , n_iterations: int):
        m = len(self.x_norm)
        tmp_theta_0 = self.theta[0]
        tmp_theta_1 = self.theta[1]
        i = 0
        self.theta_history.append([0, 0])
        while i < n_iterations:
            self.theta[0] = self.theta[0] - learning_rate * self.calculate_theta_mean(m, 0)
            self.theta[1] = self.theta[1] - learning_rate * self.calculate_theta_mean(m, 1)
            self.losses.append(self.calculate_loss())
            if abs(tmp_theta_0 - self.theta[0]) < 0.0000000001 and abs(tmp_theta_1 - self.theta[1]) < 0.0000000001:
                break   # we have reached the point of convergence
            tmp_theta_0 = self.theta[0]
            tmp_theta_1 = self.theta[1]
            if (n_iterations - i) % (n_iterations / NB_FRAME) == 0:
                self.theta_history.append([tmp_theta_0, tmp_theta_1])
            i += 1
        print(f"Theta = {self.theta}")
        return self.theta

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title("Loss over iterations")
        plt.grid(True)
        plt.show()

    def animation(self):
        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots()       
        def animate(frame_num):
            ax.clear()
            ax.scatter(self.x, self.y)
            y_pred = self.theta_history[frame_num][1] * np.array(self.x_norm, float) + self.theta_history[frame_num][0]
            ax.plot(self.x, y_pred, c='r')
            ax.set_xlim(0, max(self.x) + max(self.x) * 15 / 100)
            ax.set_ylim(0, max(self.y) + max(self.y) * 15 / 100)
        ani = FuncAnimation(fig, animate, frames=len(self.theta_history), interval=50)
        plt.close()
        ani.save("animation.gif", dpi=300, writer=PillowWriter(fps=1))