
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

NTHREADS = [2, 4, 8, 10, 16, 20]

def parse_log_file(log_file):
    data = {"ORIGINAL": {}, "METIS": {}}
    current_case = None
    current_matrix = None

    with open(log_file, 'r') as file:
        for line in file:
            if "CASE-1: ORIGINAL" in line:
                current_case = "ORIGINAL"
            elif "CASE-2: METIS" in line:
                current_case = "METIS"
            elif "START" in line:
                current_matrix = line.split(" ")[-1].strip("-######\n")
                data[current_case][current_matrix] = {n: [] for n in NTHREADS}
            elif "NTHREAD" in line:
                try:
                    nthreads = int(line.split(": ")[1].strip())
                except ValueError:
                    print(f"Warning: Could not parse thread count in line: {line.strip()}")
                    continue
            elif "Total runtime:" in line:
                try:
                    runtime = float(line.split(":")[-1].replace('|', '').strip().split()[0])
                    data[current_case][current_matrix][nthreads].append(runtime)
                except ValueError:
                    print(f"Warning: Could not parse runtime value in line: {line.strip()}")
                    continue
            elif "END" in line:
                current_matrix = None
    return data

def plot_speedup(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for case, matrices in data.items():
        for matrix, runtimes in matrices.items():
            plt.figure(figsize=(10, 6))
            speedup_data = []
            base_runtime = np.mean(runtimes[2]) if runtimes[2] else None

            # Verifica si hay datos de speedup para esta matriz
            if base_runtime:
                for n in NTHREADS:
                    if runtimes[n]:
                        avg_runtime = np.mean(runtimes[n])
                        speedup = base_runtime / avg_runtime
                        speedup_data.append((n, speedup))
                        plt.plot(n, speedup, marker='o', label=f"{n} Threads")

                plt.xlabel('Number of Threads')
                plt.ylabel('Speedup')
                plt.title(f'Speedup for {matrix} - Case: {case}')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(output_folder, f"{os.path.basename(matrix)}_{case}_speedup.png"))
                plt.close()

                # Guardar datos de speedup en CSV
                with open(os.path.join(output_folder, f"{os.path.basename(matrix)}_{case}_speedup.csv"), 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Threads", "Speedup"])
                    writer.writerows(speedup_data)
            else:
                print(f"Ignoring matrix {matrix} in {case} due to lack of base runtime data.")

def save_runtime_data(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Guardar todos los tiempos de ejecución en un solo archivo CSV
    with open(os.path.join(output_folder, "runtime_results.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case", "Matrix", "Threads", "Average Runtime (ms)"])

        for case, matrices in data.items():
            for matrix, runtimes in matrices.items():
                for n in NTHREADS:
                    if runtimes[n]:
                        avg_runtime = np.mean(runtimes[n])
                        writer.writerow([case, matrix, n, avg_runtime])

# Asegúrate de ejecutar estas funciones en tu script principal
# log_data = parse_log_file(args.log_file)
# save_runtime_data(log_data, args.output_folder)
# plot_speedup(log_data, args.output_folder)

def main():
    parser = argparse.ArgumentParser(description="Process runtime analysis logs for ORIGINAL and METIS cases.")
    parser.add_argument('--log_file', type=str, required=True, help="Path to the benchmark results log file")
    parser.add_argument('--output_folder', type=str, default="output_results", help="Folder to save output results")
    args = parser.parse_args()

    log_data = parse_log_file(args.log_file)
    save_runtime_data(log_data, args.output_folder)
    plot_speedup(log_data, args.output_folder)

if __name__ == "__main__":
    main()

