import os
import matplotlib.pyplot as plt

# Carpeta para guardar los gráficos
if not os.path.exists('plots'):
    os.makedirs('plots')

# Diccionario para guardar datos por matriz
data = {}

file_path = "/home/paul/UNSA/directs/spike_pstrsv/exs/tools/output.log"

# Leer el archivo txt
with open(file_path, 'r') as file:
    lines = file.readlines()

current_matrix = None
current_nthreads = None
current_solver = None

# Inicializar solvers
solvers = [
    'MKL Seq. Triangular Solver',
    'MKL Par. Triangular Solver',
    'SPARSKIT Triangular Solver',
    'Parallel Triangular Solver',
    'CUDA Sparse Matrix-Vector Mult.'
]

# Recorrer las líneas del archivo
for line in lines:
    line = line.strip()

    # Detectar el nombre de la matriz
    if "######-START: Tests on Upper Triangular" in line:
        start_index = line.find("/Original/")
        end_index = line.find(".mtx", start_index) + 4
        current_matrix = line[start_index:end_index]
        data[current_matrix] = {solver: {'NTHREADS': [], 'Avg. runtimes': []} for solver in solvers}

    # Detectar el número de hilos (NTHREAD)
    elif line.startswith("NTHREAD:"):
        current_nthreads = int(line.split(":")[1].strip())

    # Detectar información del solver y el tiempo promedio
    for solver in solvers:
        if solver in line and "Avg. runtime" in line:
            avg_runtime = float(line.split('| Avg. runtime: ')[1].replace(" ms", "").strip())
            data[current_matrix][solver]['NTHREADS'].append(current_nthreads)
            data[current_matrix][solver]['Avg. runtimes'].append(avg_runtime)

# Graficar y guardar cada matriz individualmente
for matrix, solvers_data in data.items():
    plt.figure(figsize=(10, 6))

    for solver, values in solvers_data.items():
        plt.plot(values['NTHREADS'], values['Avg. runtimes'], marker='o', label=solver)

    plt.title(f"Performance for {matrix}")
    plt.xlabel('Number of Threads')
    plt.ylabel('Avg. Runtime (ms)')
    plt.legend()
    plt.grid(True)

    filename = f"plots/{matrix.replace('/', '_').replace('.mtx', '')}.png"
    plt.savefig(filename)
    plt.close()

print("Gráficas generadas y guardadas en la carpeta 'shows'.")
