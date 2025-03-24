import numpy as np

import matplotlib.pyplot as plt


# Load the numpy file
data1 = np.load("../output/line_data_0.1ell.npy", allow_pickle=True).item()


data05 = np.load("../output/line_data_0.05ell.npy", allow_pickle=True).item()
data025 = np.load("../output/line_data_0.025ell.npy", allow_pickle=True).item()


# Extract x_coords and values
x_coords1 = data1["x_coords"]
values1 = data1["values"]
x_coords05 = data05["x_coords"]
values05 = data05["values"]
x_coords025 = data025["x_coords"]
values025 = data025["values"]

# Enable LaTeX typesetting
plt.rcParams['text.usetex'] = False

# Set font sizes
plt.rcParams.update({
    'axes.labelsize': 14,   # x and y labels
    'axes.titlesize': 16,   # title
    'legend.fontsize': 14,  # legend
    'xtick.labelsize': 12,  # x-axis tick labels
    'ytick.labelsize': 12   # y-axis tick labels
})

plt.figure()
plt.plot(x_coords1, values1, label=r'$\ell=0.1$', linewidth = 2)
plt.plot(x_coords05, values05, label=r'$\ell=0.05$', linewidth = 2)
plt.plot(x_coords025, values025, label=r'$\ell=0.025$', linewidth = 2)
plt.xlabel(r"$x$")
plt.ylabel(r'$\varphi$')
plt.title(r'Cross-section at central horizontal line')
plt.legend()
plt.grid(True)
plt.show()