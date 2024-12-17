
#ifndef GPU_SPMV_H
#define GPU_SPMV_H

void gpu_csr_spmv(int m, int nnz, const double* h_D, const int* h_ID, const int* h_JD,
                  const double* h_x, double* h_b, double alpha, double beta);

#endif // GPU_SPMV_H
