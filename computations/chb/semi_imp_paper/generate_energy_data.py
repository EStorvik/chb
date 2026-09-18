from chb_splitting_ch_biot_semi_imp import splitting_ch_biot_semiimp


from rich.progress import Progress

from chb import Parameters
from pathlib import Path
import pandas as pd


def extract_from_log_data(log_data=None):
    if log_data is not None:
        total_time = sum(log_data["Computational_Time"])
        total_iterations = sum(log_data["Iterations"])
    else:
        total_time = 0
        total_iterations = 0
    return total_time, total_iterations


method = splitting_ch_biot_semiimp


energies_gamma = {}
energies_swelling = {}

gamma_params = [0.25, 0.5, 1, 2, 4]
swelling_params = [0.0625, 0.125, 0.25, 0.5]


# for g in gamma:
#     parameters = Parameters(gamma = g, swelling = 0.5)
#     log_data = method(parameters)
#     energies_gamma.update({g: log_data["Total_Energy"]})

# energies_gamma_df = pd.DataFrame(energies_gamma)

# output_path_gamma = Path("output/energies_gamma.csv")
# output_path_gamma.parent.mkdir(parents=True, exist_ok=True)
# energies_gamma_df.to_csv(output_path_gamma)


for s in swelling_params:
    parameters = Parameters(swelling=s, gamma=1.0)
    log_data = method(parameters)
    energies_swelling.update({s: log_data["Total_Energy"]})


energies_swelling_df = pd.DataFrame(energies_swelling)

output_path_swelling = Path("output/energies_swelling.csv")
output_path_swelling.parent.mkdir(parents=True, exist_ok=True)
energies_swelling_df.to_csv(output_path_swelling)
