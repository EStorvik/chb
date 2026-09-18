import pandas as pd
import matplotlib.pyplot as plt

df_gamma_iter = pd.read_csv("output/total_iterations_gamma.csv", index_col=0)
df_swelling_iter = pd.read_csv("output/total_iterations_swelling.csv", index_col=0)
df_gamma_time = pd.read_csv("output/total_times_gamma.csv", index_col=0)
df_swelling_time = pd.read_csv("output/total_times_swelling.csv", index_col=0)

df_types = {
    "gamma": {
        "Total # iterations": df_gamma_iter,
        "Total simulation time": df_gamma_time,
    },
    "swelling": {
        "Total # iterations": df_swelling_iter,
        "Total simulation time": df_swelling_time,
    },
}


colors = {
    "orange": (255, 137, 0),
    "blue": (0, 144, 188),
    "green": (0, 144, 118),
    "purple": (196, 19, 252),
    "red": (254, 54, 41),
    "yellow": (255, 205, 105),
}

color_values = [tuple(channel / 255 for channel in rgb) for rgb in colors.values()]

for type, sub_types in df_types.items():
    for counter, df in sub_types.items():

        ax = df.plot.bar(color=color_values, zorder=2)

        # Add a subtle gap between bars.
        for bar in ax.patches:
            width = bar.get_width()
            new_width = width * 0.94
            bar.set_width(new_width)
            bar.set_x(bar.get_x() + (width - new_width) / 2)

        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.55, linewidth=0.8, color="0.75", zorder=0)
        ax.set_xlabel(type)
        ax.set_ylabel(counter)
        plt.tight_layout()
        plt.show()
