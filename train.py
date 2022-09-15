from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
from utils import minmax

NB_FRAME = 10

class Train:
    """Train the model thanks to gradient descent algorithm"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.x = np.array(self.df['km'], dtype='float64').reshape(-1, 1)
        self.x_norm = minmax(self.x)
        self.y = np.array(self.df['price'], dtype='float64').reshape(-1, 1)
        self.theta = np.array([0., 0.]).reshape(-1, 1)
        self.theta_history = []
        self.losses = []
   
    def calculate_loss(self):
        """Calculates the value of loss function.
        Returns:
            J_value : has to be a float.
        """
        m = len(self.x_norm)
        X = np.hstack((np.ones((self.x_norm.shape[0], 1)), self.x_norm))
        y_hat = np.dot(X, self.theta)
        return (np.sum(np.square(y_hat - self.y)) / (m * 2))

    def gradient(self):
        """Computes a gradient vector
        Returns:
            The gradient as a numpy.array, a vector of shape 2 * 1."""
        m = self.x.shape[0]
        intercept = np.ones((m, 1))
        X = np.hstack((intercept, self.x_norm))   # transform x to fit dimensions of theta
        gradient = ((X.T).dot(X.dot(self.theta) - self.y)) / m
        return gradient
    
    def fit(self, alpha: float , n_iterations: int):
        """Fits the model to the training dataset contained in x and y.
        Returns:
            theta: has to be a numpy.array, a 2 * 1 vector.
        """    
        i = 0
        while i < n_iterations:
            self.theta = self.theta - (alpha * self.gradient())
            self.losses.append(self.calculate_loss())
            if (n_iterations - i) % (n_iterations / NB_FRAME) == 0:
                self.theta_history.append([self.theta[0], self.theta[1]])
            i += 1
        print(f"Theta = {self.theta}")
        return self.theta

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.title("Loss over iterations")
        plt.show()

    def animation(self):
        plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots()       
        def animate(frame_num):
            ax.clear()
            ax.scatter(self.x, self.y)
            y_pred = self.theta_history[frame_num][1] * np.array(self.x_norm, float) + self.theta_history[frame_num][0]
            ax.plot(self.x, y_pred, c='limegreen')
            ax.set_xlim(0, max(self.x) + max(self.x) * 15 / 100)
            ax.set_ylim(0, max(self.y) + max(self.y) * 15 / 100)
        ani = FuncAnimation(fig, animate, frames=len(self.theta_history), interval=1)
        plt.close()
        ani.save("animation.gif", dpi=300, writer=PillowWriter(fps=1))