
import random

def generate_random_triangular_mtx(filename, n, triangular_type="upper", density=0.1):
    """
    Genera una matriz triangular dispersa aleatoria y la guarda en formato .mtx.
    
    Parámetros:
    - filename: nombre del archivo de salida (string).
    - n: tamaño de la matriz (entero).
    - triangular_type: tipo de triangularidad, "upper" para triangular superior o "lower" para triangular inferior.
    - density: proporción de elementos no nulos en la matriz (float, entre 0 y 1).
    """
    elements = []

    # Generar elementos no nulos en la parte triangular especificada
    for i in range(n):
        for j in range(i, n) if triangular_type == "upper" else range(0, i + 1):
            if random.random() <= density:
                value = random.uniform(0.1, 10.0)  # Valor aleatorio entre 0.1 y 10
                elements.append((i + 1, j + 1, value))  # Matrix Market usa índices basados en 1

    # Escribir la matriz en formato Matrix Market
    with open(filename, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("%\n")
        f.write(f"{n} {n} {len(elements)}\n")  # Dimensiones y número de elementos no nulos
        
        for i, j, value in elements:
            f.write(f"{i} {j} {value}\n")

    print(f"Archivo .mtx guardado como {filename}")

# Ejemplo de uso
n = 100000  # Tamaño de la matriz
triangular_type = "lower"  # Cambia a "upper" para triangular superior
generate_random_triangular_mtx("ex.mtx", n, triangular_type)

