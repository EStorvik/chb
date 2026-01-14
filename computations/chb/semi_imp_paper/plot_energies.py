import pandas as pd
import matplotlib.pyplot as plt


path = '../output/log/chb_splitting_ch_biot_'
si = 'semi_imp_'
imp = 'imp_'


GAMMA = [4, 2, 1, 0.5,0.25]
SWELLING = [0.5, 0.25, 0.125, 0.0625]

line_styles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1))]
# Vibrant colors
colors_dark = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Same colors but faded (more transparent)
colors_light = ["#6994b2", "#fbb476", "#639663", "#c87474", "#a492b5", "#8c746f"]
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'  # or 'sans-serif'
plt.rcParams['font.size'] = 16


plt.figure()

for i, g in enumerate(GAMMA):

    df = pd.read_excel(path+si+'gamma_'+str(g)+'.xlsx')
    energy = df['Total_Energy']
    plt.plot(energy,color = colors_dark[i], label=r'$\gamma$='+f'{g}', linewidth = 2, linestyle =line_styles[i])
    print(energy)
    # df = pd.read_excel(path+imp+'gamma_'+str(g)+'.xlsx')
    # energy = df['Total_Energy']
    # plt.plot(energy, color = colors_light[i], label=r'$\gamma$='+f'{g}', linewidth = 2, linestyle =line_styles[i])

plt.legend()
plt.grid(True, alpha=0.8, linestyle=':', linewidth=0.5)
plt.ylabel('Total energy ')
plt.xlabel('Time step')
# Save as PDF (best for LaTeX)
plt.savefig('plot_energy_gamma.pdf', bbox_inches='tight', dpi=300)

plt.show()


plt.figure()

for i, xi in enumerate(SWELLING):

    df = pd.read_excel(path+si+'swelling_'+str(xi)+'.xlsx')
    energy = df['Total_Energy']
    plt.plot(energy, color = colors_dark[i], label=r'$\xi$='+f'{xi}', linewidth = 2, linestyle =line_styles[i])

plt.legend()
plt.grid(True, alpha=0.8, linestyle=':', linewidth=0.5)
plt.ylabel('Total energy ')
plt.xlabel('Time step')
# Save as PDF (best for LaTeX)
plt.savefig('plot_energy_swelling.pdf', bbox_inches='tight', dpi=300)

plt.show()
