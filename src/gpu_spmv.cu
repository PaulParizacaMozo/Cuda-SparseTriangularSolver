#include <stdio.h>
#include <mkl.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <gpu_solvers.h>
#include <omp.h>

extern "C" void gpu_csr_spmv(int m, int nnz, const double* h_D, const int* h_ID, const int* h_JD,
                             const double* h_x, double* h_b, double alpha, double beta) {
    // Inicializar cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Reservar memoria en la GPU
    double *d_D, *d_x, *d_b;
    int *d_ID, *d_JD;

    cudaMalloc((void**)&d_D, nnz * sizeof(double));
    cudaMalloc((void**)&d_ID, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_JD, nnz * sizeof(int));
    cudaMalloc((void**)&d_x, m * sizeof(double));
    cudaMalloc((void**)&d_b, m * sizeof(double));

    cudaMemcpy(d_D, h_D, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ID, h_ID, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JD, h_JD, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m * sizeof(double), cudaMemcpyHostToDevice);

    // Crear descriptores
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    cusparseCreateCsr(&matA, m, m, nnz, d_ID, d_JD, d_D,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, m, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, m, d_b, CUDA_R_64F);

    // Calcular el tamaño del buffer
    size_t bufferSize = 0;
    void* dBuffer = NULL;

    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                            CUSPARSE_SPMV_CSR_ALG1, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Realizar el cálculo
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                 CUSPARSE_SPMV_CSR_ALG1, dBuffer);

    // Copiar el resultado de vuelta al host
    cudaMemcpy(h_b, d_b, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar recursos
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(d_D);
    cudaFree(d_ID);
    cudaFree(d_JD);
    cudaFree(d_x);
    cudaFree(d_b);
}


// Kernel para la resolución triangular superior: U * x = b
__global__ void udsol_kernel(int n, const double* b, double* x, const double* D, const int* ID, const int* JD) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double t = b[row];
    for (int j = ID[row] + 1; j < ID[row + 1]; ++j) {
        t -= D[j] * x[JD[j]];
    }
    x[row] = t / D[ID[row]];
}

// Kernel para la operación D2 * x = b
__global__ void udsol2_kernel(int n, const double* b, double* x, const double* D, const int* ID, const int* JD) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    double t = b[row];
    for (int j = ID[row] + 1; j < ID[row + 1]; ++j) {
        t -= D[j] * x[JD[j]];
    }
    x[row] = t / D[ID[row]];
}

// Implementación principal de CUDA
extern "C" void spike_pstrsv_solve_cuda(const char uplo, int m, double* b, double* x, 
                                        int nthreads, const double* D, const int* ID, const int* JD) {
    if (uplo != 'U' && uplo != 'u') {
        //printf("Only upper triangular matrices are supported in this implementation.\n");
        return;
    }

    // Copiar datos a la GPU
    double *d_b, *d_x, *d_D;
    int *d_ID, *d_JD;

    cudaMalloc((void**)&d_b, m * sizeof(double));
    cudaMalloc((void**)&d_x, m * sizeof(double));
    cudaMalloc((void**)&d_D, ID[m] * sizeof(double));
    cudaMalloc((void**)&d_ID, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_JD, ID[m] * sizeof(int));

    cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, ID[m] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ID, ID, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JD, JD, ID[m] * sizeof(int), cudaMemcpyHostToDevice);

    // Configuración de los kernels
    int threadsPerBlock = 256;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    // Resolver el sistema triangular superior
    udsol_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, d_b, d_x, d_D, d_ID, d_JD);

    // Resolver D2 * x = b si es necesario
    if (nthreads > 0) {
        udsol2_kernel<<<blocksPerGrid, threadsPerBlock>>>(m, d_b, d_x, d_D, d_ID, d_JD);
    }

    // Copiar los resultados de vuelta a la CPU
    cudaMemcpy(x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria en la GPU
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_D);
    cudaFree(d_ID);
    cudaFree(d_JD);
}


// Kernel optimizado para resolver el sistema triangular superior con memoria compartida
__global__ void udsol_kernel_optimized(int m, const double* __restrict__ b, double* x, 
                                       const double* __restrict__ D, const int* __restrict__ ID, 
                                       const int* __restrict__ JD) {
    __shared__ double shared_x[256];  // Memoria compartida para reutilizar valores de x
    int tid = threadIdx.x;            // Identificador del hilo dentro del bloque
    int row = blockIdx.x * blockDim.x + tid;  // Índice de fila global

    if (row < m) {
        double t = b[row];

        // Cargar valores relevantes de x en memoria compartida para optimizar accesos
        for (int j = ID[row]; j < ID[row + 1]; ++j) {
            int col = JD[j];
            shared_x[tid] = x[col];  // Asumimos coalescencia si JD está ordenado
            __syncthreads();

            t -= D[j] * shared_x[tid];
        }

        __syncthreads();  // Sincronización para asegurar que todos los hilos terminan
        x[row] = t / D[ID[row]];  // Resolver el valor diagonal
    }
}

// Implementación principal de CUDA
extern "C" double _spike_pstrsv_solve_cuda(const char uplo, int m, double* b, double* x, 
                                         int nthreads, const double* D, const int* ID, const int* JD) {
    if (uplo != 'U' && uplo != 'u') {
        printf("Only upper triangular matrices are supported in this implementation.\n");
        return -1.0;
    }

    // Variables de GPU
    double *d_b, *d_x, *d_D;
    int *d_ID, *d_JD;

    // Variables para medir tiempo
    cudaEvent_t start, stop;
    float compute_time_ms = 0.0f;  // Tiempo en milisegundos

    // Crear eventos para medir tiempo
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Asignar memoria en la GPU
    cudaMalloc((void**)&d_b, m * sizeof(double));
    cudaMalloc((void**)&d_x, m * sizeof(double));
    cudaMalloc((void**)&d_D, ID[m] * sizeof(double));
    cudaMalloc((void**)&d_ID, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_JD, ID[m] * sizeof(int));

    // Copiar datos a la GPU (excluido del tiempo computacional)
    cudaMemcpy(d_b, b, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, D, ID[m] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ID, ID, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JD, JD, ID[m] * sizeof(int), cudaMemcpyHostToDevice);

    // Configuración optimizada de grid y bloques
    int threadsPerBlock = 256;  // Ajustado para la arquitectura Pascal
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    // Iniciar medición del tiempo computacional
    cudaEventRecord(start);

    // Ejecutar el kernel optimizado
    udsol_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(m, d_b, d_x, d_D, d_ID, d_JD);
    cudaDeviceSynchronize();  // Asegurar finalización del kernel

    // Detener medición del tiempo
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&compute_time_ms, start, stop);  // Tiempo en ms

    // Copiar resultados de vuelta a la CPU
    cudaMemcpy(x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberar memoria en GPU
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_D);
    cudaFree(d_ID);
    cudaFree(d_JD);

    // Liberar eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Retornar el tiempo computacional en segundos
    return compute_time_ms / 1000.0;  // Convertir de ms a segundos
}

__global__ void triangularMatrixVectorMultKernel(int m, double *a, int *ia, int *ja, double *x, double *b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        double sum = 0.0;
        // Iterar sobre los elementos no nulos de la fila 'row'
        for (int idx = ia[row]; idx < ia[row + 1]; ++idx) {
            int col = ja[idx];  // Índice de la columna
            sum += a[idx] * x[col];  // Realizamos la multiplicación
        }
        b[row] = sum;  // Almacenamos el resultado en 'b'
    }
}

extern "C" void triangularMatrixVectorMultCUDA(int m, double *a, int *ia, int *ja, double *x, double *b) {
    double *d_a, *d_x, *d_b;
    int *d_ia, *d_ja;

    // Tamaño de la memoria a reservar para los arreglos en la GPU
    size_t size_a = sizeof(double) * (ia[m] - ia[0]);  // Solo la cantidad de elementos no nulos
    size_t size_x = m * sizeof(double);
    size_t size_b = m * sizeof(double);
    size_t size_ia = (m + 1) * sizeof(int);  // Tamaño del índice IA
    size_t size_ja = (ia[m] - ia[0]) * sizeof(int);  // Número de valores no nulos

    // Asignar memoria en el dispositivo para los valores de la matriz, vectores y arreglos de índices
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_ia, size_ia);
    cudaMalloc(&d_ja, size_ja);

    // Copiar datos desde el host hacia el dispositivo
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ia, ia, size_ia, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ja, ja, size_ja, cudaMemcpyHostToDevice);

    // Obtener propiedades de la GPU
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    // Información relevante de la GPU
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;  // Número máximo de hilos por bloque
    int warpSize = deviceProp.warpSize;  // Tamaño del warp (32 hilos)
    int numSM = deviceProp.multiProcessorCount;  // Número de multiprocesadores

    // Establecer un tamaño de bloque óptimo
    int blockSize = maxThreadsPerBlock;  // Usamos el máximo permitido por la GPU
    int numBlocks = (m + blockSize - 1) / blockSize;  // Número de bloques necesarios

    // Lanzar el kernel
    triangularMatrixVectorMultKernel<<<numBlocks, blockSize>>>(m, d_a, d_ia, d_ja, d_x, d_b);

    // Esperar a que el kernel termine de ejecutarse
    cudaDeviceSynchronize();

    // Copiar los resultados de vuelta al host
    cudaMemcpy(b, d_b, size_b, cudaMemcpyDeviceToHost);

    // Liberar la memoria del dispositivo
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_ia);
    cudaFree(d_ja);
}
