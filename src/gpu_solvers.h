
#ifndef GPU_SOLVERS_H
#define GPU_SOLVERS_H

// Prototipos para las funciones CUDA de resolución triangular

/**
 * Solución triangular superior: U * x = b.
 * Implementación en CUDA que sigue el formato CSR.
 *
 * @param n        Número de filas.
 * @param x        Vector solución (se modifica in-place).
 * @param b        Vector del lado derecho.
 * @param A        Valores de la matriz triangular superior.
 * @param IA       Índices de inicio por fila en formato CSR.
 * @param JA       Columnas correspondientes a los valores en A.
 */
void gpu_udsol_launcher(int n, double* x, const double* b, const double* A, const int* IA, const int* JA);


/**
 * Solución triangular superior con diagonal unitaria: U * x = b.
 * Implementación en CUDA para un único vector, modificando x in-place.
 *
 * @param n        Número de filas.
 * @param x        Vector solución (se modifica in-place).
 * @param A        Valores de la matriz triangular superior (sin diagonal explícita).
 * @param IA       Índices de inicio por fila en formato CSR.
 * @param JA       Columnas correspondientes a los valores en A.
 */
void gpu_usol2_launcher(int n, double* x, const double* A, const int* IA, const int* JA);

#endif // GPU_SOLVERS_H
