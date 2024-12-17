
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <iostream>

extern "C" {
/**
 * Kernel para resolver U*x = b usando una matriz triangular superior.
 */
__global__ void udsol_kernel(int n, double* x, const double* b, const double* A, const int* IA, const int* JA) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    if (row == n - 1) { // Última fila
        x[row] = b[row] / A[IA[row]];
    } else {
        double t = b[row];
        for (int j = IA[row] + 1; j < IA[row + 1]; j++) {
            t -= A[j] * x[JA[j]];
        }
        x[row] = t / A[IA[row]];
    }
}

/**
 * Kernel para resolver U*x = b usando una matriz triangular superior con diagonal unitaria.
 */
__global__ void usol2_kernel(int n, double* x, const double* A, const int* IA, const int* JA) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double t = 0;
    for (int j = IA[row]; j < IA[row + 1]; j++) {
        t += A[j] * x[JA[j]];
    }
    x[row] -= t;
}

/**
 * Función lanzadora para `udsol_kernel` que sigue la firma de `__udsol`.
 */
void gpu_udsol_launcher(int n, double* x, const double* b, const double* A, const int* IA, const int* JA) {
    // Configuración de kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Copiar datos a GPU
    double *d_x, *d_b, *d_A;
    int *d_IA, *d_JA;

    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_b, n * sizeof(double));
    cudaMalloc((void**)&d_A, IA[n] * sizeof(double));
    cudaMalloc((void**)&d_IA, (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_JA, IA[n] * sizeof(int));

    cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, IA[n] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, IA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, IA[n] * sizeof(int), cudaMemcpyHostToDevice);

    // Ejecutar kernel
    udsol_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_b, d_A, d_IA, d_JA);

    // Copiar el resultado de vuelta
    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria de GPU
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_A);
    cudaFree(d_IA);
    cudaFree(d_JA);
}

/**
 * Función lanzadora para `usol2_kernel` que sigue la firma de `__usol2`.
 */
void gpu_usol2_launcher(int n, double* x, const double* A, const int* IA, const int* JA) {
    // Configuración de kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Copiar datos a GPU
    double *d_x, *d_A;
    int *d_IA, *d_JA;

    cudaMalloc((void**)&d_x, n * sizeof(double));
    cudaMalloc((void**)&d_A, IA[n] * sizeof(double));
    cudaMalloc((void**)&d_IA, (n + 1) * sizeof(int));
    cudaMalloc((void**)&d_JA, IA[n] * sizeof(int));

    cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, IA[n] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IA, IA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, IA[n] * sizeof(int), cudaMemcpyHostToDevice);

    // Ejecutar kernel
    usol2_kernel<<<blocksPerGrid, threadsPerBlock>>>(n, d_x, d_A, d_IA, d_JA);

    // Copiar el resultado de vuelta
    cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria de GPU
    cudaFree(d_x);
    cudaFree(d_A);
    cudaFree(d_IA);
    cudaFree(d_JA);
}
} // extern "C"

