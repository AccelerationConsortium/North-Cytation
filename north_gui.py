import numpy as np
import matplotlib.pyplot as plt
import itertools
import datetime
import slack_agent

class RealTimePlot:
    def __init__(self, num_subplots=1, styles=None):
        """
        Initializes the real-time plot with horizontal subplots.

        Parameters:
        - num_subplots (int): Number of subplots in the figure.
        - styles (list of dict): List of customization options for each subplot.
        """
        self.num_subplots = num_subplots
        self.styles = styles if styles else [{} for _ in range(num_subplots)]
        self.colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  # Cycle through colors

        # Create horizontal layout (1 row, num_subplots columns)
        self.fig, self.axes = plt.subplots(1, num_subplots, figsize=(6*num_subplots, 4))  
        if num_subplots == 1:
            self.axes = [self.axes]  # Ensure it's iterable

        # Store multiple data series per subplot
        self.lines = [[] for _ in range(num_subplots)]  # List of lists for multiple series

        for i, ax in enumerate(self.axes):
            ax.set_xlim(300, 800)  # Initial x-axis range
            ax.set_ylim(0, 3)  # Initial y-axis range
            ax.set_title(f"Plot {i+1}")

        plt.show(block=False)  # No blocking for updates
        self.fig.canvas.flush_events()  # Ensure initial drawing is flushed

    def add_data(self, subplot_index, x_data, y_data, plot_type='-',color=None):
        """
        Adds a new data series to the specified subplot with a different color.
        Optionally, you can plot points or lines by setting plot_type.
        """
        if subplot_index < 0 or subplot_index >= self.num_subplots:
            raise IndexError("Invalid subplot index")

        if color is None:
            color = next(self.colors)  # Pick a new color
        print(f"Adding new series to subplot {subplot_index} with color {color} and plot type {plot_type}")

        # Choose between plotting points ('o') or lines ('-')
        line, = self.axes[subplot_index].plot(x_data, y_data, plot_type, color=color)
        self.lines[subplot_index].append(line)

        # 🔹 **Fix: Ensure autoscaling after new data is added**
        x_all = np.concatenate([line.get_xdata() for line in self.lines[subplot_index]])
        y_all = np.concatenate([line.get_ydata() for line in self.lines[subplot_index]])

        if len(x_all) > 0 and len(y_all) > 0:  # Avoid errors if empty
            self.axes[subplot_index].set_xlim(min(x_all), max(x_all))
            self.axes[subplot_index].set_ylim(min(y_all), max(y_all))

        # Only redraw the canvas when new data is added
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show(self):
        """
        Displays the plot (blocks execution).
        """
        plt.show()

    def save_figure(self):
        """
        Saves the current figure with a filename based on the current date and time.
        """
        # Get the current date and time
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create a filename with the current date and time
        filename = f"plot_{current_time}.png"
        
        # Save the figure
        self.fig.savefig(filename)
        print(f"Figure saved as {filename}")
        return filename
    
    def upload_file(filename):
        slack_agent.upload_and_post_file(filename)
