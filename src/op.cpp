#include<common.h>

void spmm_forward(
  torch::Tensor& output,
  vector<torch::Tensor>& input, 
    torch::Tensor& row_pointers, 
    torch::Tensor& column_index, 
    const nidType nodePerPE,
    const nidType numNodes, 
    const int dim,
    const int partSize,
    const int warpPerBlock,
    const int currGPUid){
    cudaSetDevice(currGPUid);
    spmm_cuda(
      output, 
      input, 
      row_pointers, 
      column_index, 
      nodePerPE,
      numNodes, 
      dim,
      partSize,
      warpPerBlock,
      currGPUid);
}

void uvm_test(torch::Tensor& t1, vector<torch::Tensor>& vt){
    print(t1);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_forward", &spmm_forward, "spmm_forward");
  m.def("uvm_test",&uvm_test, "uvm_test");
}