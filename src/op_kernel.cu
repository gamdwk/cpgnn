#include <common.h>
#include <vector>
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void spmm_uvm_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<nidType,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<nidType,1,torch::RestrictPtrTraits> column_index,
    int part_id,
    int numNodes, int partSize, int warpPerBlock
    );
template <typename scalar_t>
__global__ void spmm_uvm_cuda_kernel();

__global__ void warmup(){}

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
){
    const int lb_src =  nodePerPE * currGPUid;
    const int ub_src = min(lb_src+nodePerPE, numNodes);

    const nidType block = warpPerBlock * WARP_SIZE;
    const nidType grid = ub_src - lb_src;
    const int shared_memory_size = warpPerBlock * dim * sizeof(float) + warpPerBlock * partSize * sizeof(nidType);
    
    float ** inputs_ptr=nullptr;
    cudaMalloc(&inputs_ptr, sizeof(float*)*numNodes);
    for(int i=0;i<input.size();++i){
        inputs_ptr[i] = input[i].data_ptr<float>();
    }
    SAG_UVM_updated_cuda_kernel<<<grid, block, shared_memory_size>>>(
        output.data_ptr<float>(),
        inputs_ptr,
        row_pointers.data_ptr<nidType>(),
        column_index.data_ptr<nidType>(),
        nodePerPE,
        numNodes, 
        dim,
        partSize,
        warpPerBlock,
        currGPUid
    );
}

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
)
{
    nidType srcId_local = blockIdx.x;
    nidType srcId = blockIdx.x + currGPUid * nodePerPE;             // global node id.
    nidType block_warpId = threadIdx.x / WARP_SIZE;                 // block warp-id
    nidType laneid = threadIdx.x % WARP_SIZE;                       // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                  // part information.
    // nidType* warp_nbs = (nidType*)&part_meta[warpPerBlock*dim];        // cache neighbor id (warpPerBlock*partsize)

    if (srcId < numNodes){

        const nidType neighborBeg = row_pointers[srcId];        // partitioning pointer start
        const nidType neighborEnd = row_pointers[srcId + 1];    // part pointer end

        __syncwarp();

        for (nidType nidx_b = neighborBeg; nidx_b < neighborEnd; nidx_b += partSize*warpPerBlock){

            const nidType w_start = nidx_b + partSize * block_warpId;
            const nidType w_end = w_start + partSize < neighborEnd?  w_start + partSize: neighborEnd;

            __syncwarp();

            for(nidType nidx = 0; nidx < w_end - w_start; nidx++){  
                nidType nid = column_index[w_start + nidx];
                nidType gpuid = nid / nodePerPE;
                nidType gpu_local_nid = nid % nodePerPE;

                for (int d = laneid; d < dim; d += 32){
                    atomicAdd((float*)&output[srcId_local * dim + d], input[gpuid][gpu_local_nid * dim + d]);
                }
            }
        }
    } 
}

