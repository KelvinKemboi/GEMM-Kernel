#include <cuda_runtime.h>

__global__ void matrixMultiply(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum; // Perform matrix multiplication
    }
}

int main() {
    const int N = 1 << 10; // Size of matrices (1024 x 1024)
    size_t size = N * N * sizeof(float);

    // Allocate host memory - CPU
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialize input matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = static_cast<float>(i + j);
            h_B[i * N + j] = static_cast<float>(i - j);
        }
    }

    // Allocate device memory - GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with enough blocks and threads
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result matrix from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float expected = 0.0f;
            for (int k = 0; k < N; ++k) {
                expected += h_A[i * N + k] * h_B[k * N + j];
            }
            if (h_C[i * N + j] != expected) {
                std::cerr << "Error at (" << i << ", " << j << "): " << h_C[i * N + j] << " != " << expected << std::endl;
                success = false;
                break;
            }
        }
        if (!success) break;
    }      
} 
