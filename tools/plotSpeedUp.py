
import argparse
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Constante para los números de hilos
NTHREADS = [2, 4, 8, 10, 16, 20]

# Función para parsear el archivo de log y generar datos de tiempos
def parse_log_file(log_file):
    data = defaultdict(lambda: defaultdict(dict))
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

# Función para guardar los datos de tiempos de ejecución en un archivo CSV
def save_runtime_data(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_file = os.path.join(output_folder, "runtime_results.csv")
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case", "Matrix", "Threads", "Average Runtime (ms)"])

        for case, matrices in data.items():
            for matrix, runtimes in matrices.items():
                for n in NTHREADS:
                    if runtimes[n]:
                        avg_runtime = np.mean(runtimes[n])
                        writer.writerow([case, matrix, n, avg_runtime])
    return csv_file

# Función para leer los datos del archivo CSV
def read_runtime_data(csv_file):
    data = defaultdict(lambda: defaultdict(dict))
    
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            case = row['Case']
            matrix = row['Matrix']
            threads = int(row['Threads'])
            avg_runtime = float(row['Average Runtime (ms)'])
            data[case][matrix][threads] = avg_runtime
    return data

# Función para calcular el mejor speedup
def calculate_best_speedup(data):
    best_speedups = {}
    
    for matrix in data['ORIGINAL']:
        best_pstrsv = {'speedup': 0, 'threads': 0, 'case': ''}
        best_mkl = {'speedup': 0, 'threads': 0, 'case': ''}

        # Toma el tiempo secuencial (de 2 threads) como base para calcular el speedup
        if 2 in data['ORIGINAL'][matrix]:
            base_runtime = data['ORIGINAL'][matrix][2]

            for case in data:
                for threads, runtime in data[case][matrix].items():
                    speedup = base_runtime / runtime
                    if 'PSTRSV' in case and speedup > best_pstrsv['speedup']:
                        best_pstrsv.update({'speedup': speedup, 'threads': threads, 'case': case[0]})
                    elif 'MKL' in case and speedup > best_mkl['speedup']:
                        best_mkl.update({'speedup': speedup, 'threads': threads, 'case': case[0]})

            best_speedups[matrix] = {'PSTRSV': best_pstrsv, 'MKL': best_mkl}
    
    return best_speedups

# Función para graficar los mejores speedups
def plot_best_speedup(best_speedups, output_file):
    matrices = list(best_speedups.keys())
    pstrsv_speedups = [best_speedups[matrix]['PSTRSV']['speedup'] for matrix in matrices]
    mkl_speedups = [best_speedups[matrix]['MKL']['speedup'] for matrix in matrices]

    x = np.arange(len(matrices))
    width = 0.35  # Ancho de las barras

    fig, ax = plt.subplots(figsize=(15, 7))
    bars_pstrsv = ax.bar(x - width/2, pstrsv_speedups, width, label='PSTRSV', color='salmon')
    bars_mkl = ax.bar(x + width/2, mkl_speedups, width, label='MKL', color='steelblue')

    # Etiquetas de reordenamiento y hilos en las barras
    for i, matrix in enumerate(matrices):
        pstrsv_info = best_speedups[matrix]['PSTRSV']
        mkl_info = best_speedups[matrix]['MKL']

        ax.text(x[i] - width/2, pstrsv_info['speedup'] + 0.1,
                f"{pstrsv_info['case']}\n{pstrsv_info['threads']}",
                ha='center', va='bottom', fontsize=8)

        ax.text(x[i] + width/2, mkl_info['speedup'] + 0.1,
                f"{mkl_info['case']}\n{mkl_info['threads']}",
                ha='center', va='bottom', fontsize=8)

    # Configuración del gráfico
    ax.set_xlabel('Matrices')
    ax.set_ylabel('Speed-up')
    ax.set_title('Mejor Speed-up alcanzado por PSTRSV y MKL para cada matriz')
    ax.set_xticks(x)
    ax.set_xticklabels([matrix.split('/')[-1].replace('.mtx', '') for matrix in matrices], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Función principal para ejecutar el script
def main():
    parser = argparse.ArgumentParser(description="Proceso de log para obtener CSV y graficar el mejor speedup.")
    parser.add_argument('--log_file', type=str, required=True, help="Ruta al archivo de log de resultados.")
    parser.add_argument('--output_folder', type=str, required=True, help="Carpeta para guardar los resultados y gráficos.")
    args = parser.parse_args()

    # Parsear el archivo de log y guardar datos en CSV
    log_data = parse_log_file(args.log_file)
    csv_file = save_runtime_data(log_data, args.output_folder)

    # Leer el CSV y calcular el mejor speedup
    runtime_data = read_runtime_data(csv_file)
    best_speedups = calculate_best_speedup(runtime_data)

    # Graficar los mejores speedups
    output_image = os.path.join(args.output_folder, "best_speedup_comparison.png")
    plot_best_speedup(best_speedups, output_image)

if __name__ == "__main__":
    main()
