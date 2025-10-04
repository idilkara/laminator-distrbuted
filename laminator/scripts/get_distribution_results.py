# Authors: Vasisht Duddu, Oskari Järvinen, Lachlan J Gunn, N Asokan
# Copyright 2025 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import numpy as np

columns_dist = ["dataset", "epochs", "model_architecture", "attestation_type", "with_sgx?", "preprocess_time", "input_load_time", "input_measure_time", "compute_time", "output_measurement_time", "output_storage_time", "attestation_time", "compute_race_z", "compute_sex_z", "compute_race_z_y", "compute_sex_z_y"]

census_preproc_list, census_input_load_list, census_input_meas_list, census_compute_list, census_attestation_list, census_output_storage_list, census_output_meas_list, census_compute_race_z_list, census_compute_sex_z_list, census_compute_race_zy_list, census_compute_sex_zy_list = [], [], [], [], [], [], [], [], [], [], []
utkface_preproc_list, utkface_input_load_list, utkface_input_meas_list, utkface_compute_list, utkface_attestation_list, utkface_output_storage_list, utkface_output_meas_list, utkface_compute_race_z_list, utkface_compute_sex_z_list, utkface_compute_race_zy_list, utkface_compute_sex_zy_list = [], [], [], [], [], [], [], [], [], [], []
for run_index in range(1,2):
    with open(f'distribution_results_run{run_index}.csv', mode ='r') as file:
        csvFile = csv.reader(file)
        census_preproc_list_temp, census_input_load_list_temp, census_input_meas_list_temp, census_compute_list_temp, census_attestation_list_temp, census_output_storage_list_temp, census_output_meas_list_temp, census_compute_race_z_list_temp, census_compute_sex_z_list_temp, census_compute_race_zy_list_temp, census_compute_sex_zy_list_temp = [], [], [], [], [], [], [], [], [], [], []
        utkface_preproc_list_temp, utkface_input_load_list_temp, utkface_input_meas_list_temp, utkface_compute_list_temp, utkface_attestation_list_temp, utkface_output_storage_list_temp, utkface_output_meas_list_temp, utkface_compute_race_z_list_temp, utkface_compute_sex_z_list_temp, utkface_compute_race_zy_list_temp, utkface_compute_sex_zy_list_temp = [], [], [], [], [], [], [], [], [], [], []
        for lines in csvFile:
            preprocess_time = float(lines[5])
            input_load_time = float(lines[6])
            input_measure_time = float(lines[7])
            compute_time = float(lines[8])
            output_measurement_time = float(lines[9])
            output_storage_time = float(lines[10])
            attestation_time = float(lines[11])
            compute_race_z = float(lines[12])
            compute_sex_z = float(lines[13])
            compute_race_zy = float(lines[14])
            compute_sex_zy = float(lines[15])

            if lines[0] == "CENSUS":
                census_preproc_list_temp.append(preprocess_time)
                census_input_load_list_temp.append(input_load_time) 
                census_input_meas_list_temp.append(input_measure_time)
                census_compute_list_temp.append(compute_time)
                census_attestation_list_temp.append(attestation_time)
                census_output_storage_list_temp.append(output_storage_time)
                census_output_meas_list_temp.append(output_measurement_time)
                census_compute_race_z_list_temp.append(compute_race_z)
                census_compute_sex_z_list_temp.append(compute_sex_z)
                census_compute_race_zy_list_temp.append(compute_race_zy)
                census_compute_sex_zy_list_temp.append(compute_sex_zy)

            else:
                utkface_preproc_list_temp.append(preprocess_time)
                utkface_input_load_list_temp.append(input_load_time) 
                utkface_input_meas_list_temp.append(input_measure_time)
                utkface_compute_list_temp.append(compute_time)
                utkface_attestation_list_temp.append(attestation_time)
                utkface_output_storage_list_temp.append(output_storage_time)
                utkface_output_meas_list_temp.append(output_measurement_time)
                utkface_compute_race_z_list_temp.append(compute_race_z)
                utkface_compute_sex_z_list_temp.append(compute_sex_z)
                utkface_compute_race_zy_list_temp.append(compute_race_zy)
                utkface_compute_sex_zy_list_temp.append(compute_sex_zy)

    census_preproc_list.append(census_preproc_list_temp)
    census_input_load_list.append(census_input_load_list_temp) 
    census_input_meas_list.append(census_input_meas_list_temp)
    census_compute_list.append(census_compute_list_temp)
    census_attestation_list.append(census_attestation_list_temp)
    census_output_storage_list.append(census_output_storage_list_temp)
    census_output_meas_list.append(census_output_meas_list_temp)
    census_compute_race_z_list.append(census_compute_race_z_list_temp)
    census_compute_sex_z_list.append(census_compute_sex_z_list_temp)
    census_compute_race_zy_list.append(census_compute_race_zy_list_temp)
    census_compute_sex_zy_list.append(census_compute_sex_zy_list_temp)

    utkface_preproc_list.append(utkface_preproc_list_temp)
    utkface_input_load_list.append(utkface_input_load_list_temp) 
    utkface_input_meas_list.append(utkface_input_meas_list_temp)
    utkface_compute_list.append(utkface_compute_list_temp)
    utkface_attestation_list.append(utkface_attestation_list_temp)
    utkface_output_storage_list.append(utkface_output_storage_list_temp)
    utkface_output_meas_list.append(utkface_output_meas_list_temp)
    utkface_compute_race_z_list.append(utkface_compute_race_z_list_temp)
    utkface_compute_sex_z_list.append(utkface_compute_sex_z_list_temp)
    utkface_compute_race_zy_list.append(utkface_compute_race_zy_list_temp)
    utkface_compute_sex_zy_list.append(utkface_compute_sex_zy_list_temp)


census_preproc_list, census_input_load_list, census_input_meas_list, census_compute_list, census_attestation_list, census_output_storage_list, census_output_meas_list, census_compute_race_z_list, census_compute_sex_z_list, census_compute_race_zy_list, census_compute_sex_zy_list = map(np.array, [census_preproc_list, census_input_load_list, census_input_meas_list, census_compute_list, census_attestation_list, census_output_storage_list, census_output_meas_list, census_compute_race_z_list, census_compute_sex_z_list, census_compute_race_zy_list, census_compute_sex_zy_list])
utkface_preproc_list, utkface_input_load_list, utkface_input_meas_list, utkface_compute_list, utkface_attestation_list, utkface_output_storage_list, utkface_output_meas_list, utkface_compute_race_z_list, utkface_compute_sex_z_list, utkface_compute_race_zy_list, utkface_compute_sex_zy_list = map(np.array, [utkface_preproc_list, utkface_input_load_list, utkface_input_meas_list, utkface_compute_list, utkface_attestation_list, utkface_output_storage_list, utkface_output_meas_list, utkface_compute_race_z_list, utkface_compute_sex_z_list, utkface_compute_race_zy_list, utkface_compute_sex_zy_list])

# computing baseline 
census_baseline = census_preproc_list + census_compute_race_z_list + census_compute_sex_z_list + census_compute_race_zy_list + census_compute_sex_zy_list + census_output_storage_list
census_overhead = census_input_meas_list + census_output_meas_list + census_attestation_list
utkface_baseline = utkface_preproc_list + utkface_compute_race_z_list + utkface_compute_sex_z_list + utkface_compute_race_zy_list + utkface_compute_sex_zy_list + utkface_output_storage_list
utkface_overhead = utkface_input_meas_list + utkface_output_meas_list + utkface_attestation_list

print("\\textbf{Preprocessing}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_preproc_list,axis=0)[1]:.2f} $\pm$ {np.std(census_preproc_list,axis=0)[1]:.2f} s" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_preproc_list,axis=0)[1]:.2f} $\pm$ {np.std(utkface_preproc_list,axis=0)[1]:.2f} s\\\\")
print("\\textbf{Input Load}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_input_load_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_input_load_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_input_load_list,axis=0)[1]:.2f} $\pm$ {np.std(utkface_input_load_list,axis=0)[1]:.2f} s\\\\")
print("\\textbf{Computation (Race: z)}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_compute_race_z_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_compute_race_z_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_compute_race_z_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_compute_race_z_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\\textbf{Computation (Sex: z)}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_compute_sex_z_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_compute_sex_z_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_compute_sex_z_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_compute_sex_z_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\\textbf{Computation (Race: z|y)}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_compute_race_zy_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_compute_race_zy_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_compute_race_zy_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_compute_race_zy_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\\textbf{Computation (Sex: z|y)}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_compute_sex_zy_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_compute_sex_zy_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_compute_sex_zy_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_compute_sex_zy_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\\textbf{Output Storage}" + " & \cellcolor{gray!15}"+ f"{np.mean(census_output_storage_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_output_storage_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{gray!15}"+ f"{np.mean(utkface_output_storage_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_output_storage_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\midrule")
print("\\textbf{Baseline}" + " & \cellcolor{orange!15}"+ f"{np.mean(census_baseline,axis=0)[1]:.2f} $\pm$ {np.std(census_baseline,axis=0)[1]:.2f} s" + " & \cellcolor{orange!15}"+ f"{np.mean(utkface_baseline,axis=0)[1]:.2f} $\pm$ {np.std(utkface_baseline,axis=0)[1]:.2f} s\\\\")
print("\midrule")
print("\\textbf{Input Measurement}" +  " & \cellcolor{blue!15}"+ f"{np.mean(census_input_meas_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_input_meas_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{blue!15}"+ f"{np.mean(utkface_input_meas_list,axis=0)[1]:.2f} $\pm$ {np.std(utkface_input_meas_list,axis=0)[1]:.2f} s\\\\")
print("\\textbf{Output Measurement}" +  " & \cellcolor{blue!15}"+ f"{np.mean(census_output_meas_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_output_meas_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{blue!15}"+ f"{np.mean(utkface_output_meas_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_output_meas_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\\textbf{Attestation}" +  " & \cellcolor{blue!15}"+ f"{np.mean(census_attestation_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_attestation_list,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{blue!15}"+ f"{np.mean(utkface_attestation_list,axis=0)[1]*1000:.2f} $\pm$ {np.std(utkface_attestation_list,axis=0)[1]*1000:.2f} ms\\\\")
print("\midrule")
print("\\textbf{Overhead}"  + " & \cellcolor{orange!15}"+ f"{np.mean(census_overhead,axis=0)[1]*1000:.2f} $\pm$ {np.std(census_overhead,axis=0)[1]*1000:.2f} ms" + " & \cellcolor{orange!15}"+ f"{np.mean(utkface_overhead,axis=0)[1]:.2f} $\pm$ {np.std(utkface_overhead,axis=0)[1]:.2f} s\\\\")
print("\\textbf{Overhead (\%)}"  + " & \cellcolor{orange!15}"+ f"{np.mean(census_overhead,axis=0)[1]/np.mean(census_baseline,axis=0)[1]*100:.2f}" + " & \cellcolor{orange!15}"+ f"{np.mean(utkface_overhead,axis=0)[1]/np.mean(utkface_baseline,axis=0)[0]*100:.2f}\\\\")
