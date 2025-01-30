import numpy as np
import matplotlib.pyplot as plt
import time

class RealTimePlot:
    def __init__(self, num_subplots=1, styles=None):
        """
        Initializes the real-time plot with horizontal subplots.

        Parameters:
        - num_subplots (int): Number of subplots in the figure.
        - update_interval (int): Interval (in ms) for updating the plots.
        - styles (list of dict): List of customization options for each subplot.
        """
        self.num_subplots = num_subplots
        self.styles = styles if styles else [{} for _ in range(num_subplots)]
        
        # Create horizontal layout (1 row, num_subplots columns)
        self.fig, self.axes = plt.subplots(1, num_subplots, figsize=(8*num_subplots, 5))  
        if num_subplots == 1:
            self.axes = [self.axes]  # Ensure it's iterable

        # Store data for each subplot
        self.x_data = [[] for _ in range(num_subplots)]
        self.y_data = [[] for _ in range(num_subplots)]
        self.lines = []

        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], **self.styles[i])  # Apply style
            self.lines.append(line)
            ax.set_xlim(0, 10)  # Initial x-axis range
            ax.set_ylim(-1, 1)  # Initial y-axis range
            ax.set_title(f"Plot {i+1}")

        # Enable interactive mode
        plt.ion()
        plt.show()

    def add_data(self, subplot_index, x_data, y_data):
        """
        Adds data to the specified subplot.

        Parameters:
        - subplot_index (int): Index of the subplot to update.
        - x (float): New x-value.
        - y (float): New y-value.
        """

        for i in range (0, len(x_data)):
            x = x_data[i]
            y = y_data[i]
            if subplot_index < 0 or subplot_index >= self.num_subplots:
                raise IndexError("Invalid subplot index")

            self.x_data[subplot_index].append(x)
            self.y_data[subplot_index].append(y)

            # Keep only recent data for real-time effect
            if len(self.x_data[subplot_index]) > 100:
                self.x_data[subplot_index].pop(0)
                self.y_data[subplot_index].pop(0)

            # Update plot immediately
            self.lines[subplot_index].set_data(self.x_data[subplot_index], self.y_data[subplot_index])
            self.axes[subplot_index].relim()
            self.axes[subplot_index].autoscale_view()
            
            plt.pause(0.001)  # Force real-time update

    def show(self):
        """
        Displays the plot (blocks execution).
        """
        plt.ioff()
        plt.show()