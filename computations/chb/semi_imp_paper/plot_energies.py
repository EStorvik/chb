import pandas as pd
import matplotlib.pyplot as plt

path = "output/energies_"
gamma_path = path + "gamma.csv"
swelling_path = path + "swelling.csv"

energies_gamma = pd.read_csv(gamma_path, index_col=0)
energies_swelling = pd.read_csv(swelling_path, index_col=0)

line_styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
# Vibrant colors
colors_dark = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Same colors but faded (more transparent)
colors_light = ["#6994b2", "#fbb476", "#639663", "#c87474", "#a492b5", "#8c746f"]
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # or 'sans-serif'
plt.rcParams["font.size"] = 16


plt.figure()
i = 0
for g, energy in reversed(list(energies_gamma.items())):

    plt.plot(
        energy,
        color=colors_dark[i],
        label=r"$\gamma$=" + f"{g}",
        linewidth=2,
        linestyle=line_styles[i],
    )
    i += 1
    # df = pd.read_excel(path+imp+'gamma_'+str(g)+'.xlsx')
    # energy = df['Total_Energy']
    # plt.plot(energy, color = colors_light[i], label=r'$\gamma$='+f'{g}', linewidth = 2, linestyle =line_styles[i])

plt.legend()
plt.grid(True, alpha=0.8, linestyle=":", linewidth=0.5)
plt.ylabel("Total energy ")
plt.xlabel("Time step")
# Save as PDF (best for LaTeX)
# plt.savefig("plot_energy_gamma.pdf", bbox_inches="tight", dpi=300)

plt.show()


plt.figure()
i = 0

for s, energy in reversed(list(energies_swelling.items())):

    plt.plot(
        energy,
        color=colors_dark[i],
        label=r"$\xi$=" + f"{s}",
        linewidth=2,
        linestyle=line_styles[i],
    )
    i += 1
plt.legend()
plt.grid(True, alpha=0.8, linestyle=":", linewidth=0.5)
plt.ylabel("Total energy ")
plt.xlabel("Time step")
# Save as PDF (best for LaTeX)
plt.savefig("plot_energy_swelling.pdf", bbox_inches="tight", dpi=300)

plt.show()
