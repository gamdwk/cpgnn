#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
using namespace std;
using nidType = int;

void spmm_cuda(torch::Tensor& output, 
    vector<torch::Tensor>& input, 
    torch::Tensor& row_pointers, 
    torch::Tensor& column_index, 
    const nidType nodePerPE,
    const nidType numNodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock,
    const int currGPUid
);
__global__ 
void SAG_UVM_updated_cuda_kernel(
    float*  output,
    float** input,
    const nidType* row_pointers, 
    const nidType* column_index,
    const nidType nodePerPE,
    const nidType numNodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock,
    const int currGPUid
);