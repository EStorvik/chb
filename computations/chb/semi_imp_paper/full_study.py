

from chb_monolithic_imp import monolithic_imp
from chb_monolithic_semi_imp import monolithic_semi_imp
from chb_splitting_ch_biot_imp import splitting_ch_biot_imp
from chb_splitting_ch_biot_semi_imp import splitting_ch_biot_semiimp
from chb_splitting_ch_fixed_stress_imp import splitting_ch_fs_imp
from chb_splitting_ch_fixed_stress_semi_imp import splitting_ch_fs_semiimp

from rich.progress import Progress

from chb import Parameters
from pathlib import Path
import pandas as pd

def extract_from_log_data(log_data = None):
    if log_data is not None:
        total_time = sum(log_data["Computational_Time"])
        total_iterations = sum(log_data["Iterations"])
    else:
        total_time = 0
        total_iterations = 0
    return total_time, total_iterations


methods = {
    "Monolithic semi-implicit": monolithic_semi_imp,
    "Monolithic implicit": monolithic_imp,
    "Splitting ch-biot semi-implicit": splitting_ch_biot_semiimp,
    "Splitting ch-biot implicit": splitting_ch_biot_imp,
    "Splitting ch-fs semi-implicit": splitting_ch_fs_semiimp,
    "Splitting ch-fs implicit": splitting_ch_fs_imp,
}


total_time_data_gamma = {
    "Monolithic semi-implicit": [],
    "Monolithic implicit": [],
    "Splitting ch-biot semi-implicit": [],
    "Splitting ch-biot implicit": [],
    "Splitting ch-fs semi-implicit": [],
    "Splitting ch-fs implicit": [],
}

total_iteration_data_gamma = {
    "Monolithic semi-implicit": [],
    "Monolithic implicit": [],
    "Splitting ch-biot semi-implicit": [],
    "Splitting ch-biot implicit": [],
    "Splitting ch-fs semi-implicit": [],
    "Splitting ch-fs implicit": [],
}

total_time_data_swelling = {
    "Monolithic semi-implicit": [],
    "Monolithic implicit": [],
    "Splitting ch-biot semi-implicit": [],
    "Splitting ch-biot implicit": [],
    "Splitting ch-fs semi-implicit": [],
    "Splitting ch-fs implicit": [],
}

total_iteration_data_swelling = {
    "Monolithic semi-implicit": [],
    "Monolithic implicit": [],
    "Splitting ch-biot semi-implicit": [],
    "Splitting ch-biot implicit": [],
    "Splitting ch-fs semi-implicit": [],
    "Splitting ch-fs implicit": [],
}

nx = 64
ny = 64
ell = 0.025
gamma_values = [0.25, 0.5, 1, 2, 4]
swelling_values = [0.0625, 0.125, 0.25, 0.5]

with Progress() as p:
    t = p.add_task("Running all combinations of simulations...", total = (len(gamma_values) + len(swelling_values))*6)
    for gamma in gamma_values:
        parameters = Parameters(nx = nx, ny = ny, ell = ell, gamma = gamma, swelling = 0.5)

        for name, strategy in methods.items():
            try:
                log_data = strategy(parameters)
                print(f"{name} at gamma = {gamma} successfully converged")

            except:
                print(f"{name} at gamma = {gamma} failed to converge")
                log_data = None
            total_time, total_iterations = extract_from_log_data(log_data)


            total_time_data_gamma[name].append(total_time)
            total_iteration_data_gamma[name].append(total_iterations)
            p.update(t, advance=1)

    for swelling in swelling_values:
        parameters = Parameters(nx = nx, ny = ny, ell = ell, gamma = 1, swelling = swelling)
        for name, strategy in methods.items():
            try:
                log_data = strategy(parameters)
                print(f"{name} at swelling = {swelling} successfully converged")

            except:
                print(f"{name} at swelling = {swelling} failed to converge")
                log_data = None
            total_time, total_iterations = extract_from_log_data(log_data)

            total_time_data_swelling[name].append(total_time)
            total_iteration_data_swelling[name].append(total_iterations)

            p.update(t, advance= 1)

time_df_gamma = pd.DataFrame(total_time_data_gamma, index = gamma_values)
time_output_path_gamma = Path("output/total_times_gamma.csv")
time_output_path_gamma.parent.mkdir(parents = True, exist_ok = True)
time_df_gamma.to_csv(time_output_path_gamma)

iteration_df_gamma = pd.DataFrame(total_iteration_data_gamma, index = gamma_values)
iteration_output_path_gamma = Path("output/total_iterations_gamma.csv")
iteration_output_path_gamma.parent.mkdir(parents = True, exist_ok = True)
iteration_df_gamma.to_csv(iteration_output_path_gamma)

time_df_swelling = pd.DataFrame(total_time_data_swelling, index = swelling_values)
time_output_path_swelling = Path("output/total_times_swelling.csv")
time_output_path_swelling.parent.mkdir(parents = True, exist_ok = True)
time_df_swelling.to_csv(time_output_path_swelling)

iteration_df_swelling = pd.DataFrame(total_iteration_data_swelling, index = swelling_values)
iteration_output_path_swelling = Path("output/total_iterations_swelling.csv")
iteration_output_path_swelling.parent.mkdir(parents = True, exist_ok = True)
iteration_df_swelling.to_csv(iteration_output_path_swelling)




