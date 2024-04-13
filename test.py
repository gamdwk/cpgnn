import torch
import cpgnn 
import fbgemm_gpu
#import dgl
import torch.multiprocessing as mp
import networkx as nx
import random
import numpy as np
import scipy.sparse as sp

def generate_random_digraph(nodes, edges):
    # 创建一个有向图
    G = nx.DiGraph()

    # 添加节点
    for node in range(nodes):
        G.add_node(node)

    # 添加边
    for _ in range(edges):
        source = random.randint(0, nodes - 1)  # 随机选择源节点
        target = random.randint(0, nodes - 1)  # 随机选择目标节点
        if source != target and not G.has_edge(source, target):
            # 避免自环和重复边
            G.add_edge(source, target)

    return G

def getCSR(graph):
    adjacency_matrix = nx.adjacency_matrix(graph)

    # 将邻接矩阵转换为CSR格式
    csr_matrix = sp.csr_matrix(adjacency_matrix)

    # 计算行指针数组
    row_ptr = csr_matrix.indptr
    column_index = csr_matrix.indices

    return torch.tensor(row_ptr, dtype=torch.int32), torch.tensor(column_index, dtype=torch.int32)


def test(output, 
      input_tensor, 
      row_pointers, 
      column_index, 
      nodePerPE,
      numNodes, 
      dim,
      partSize,
      warpPerBlock,
      currGPUid):
    print("rank" , currGPUid ,"start")
    cpgnn.spmm_forward(
        output, 
      input_tensor, 
      row_pointers, 
      column_index, 
      int(nodePerPE),
      numNodes, 
      dim,
      partSize,
      warpPerBlock,
      currGPUid
    )
    torch.cuda.synchronize()
    print("finish")
    
if __name__ == "__main__":
    num_nodes = 100
    num_edges = 5000
    dim = 100
    num_gpu = 2
    nodePerGpu = num_nodes / 2;
    gpu_tensor = torch.rand((num_nodes, dim))
    gpu_inputs = []
    unifi_input = []
    unifi_output = []
    
    start = 0
    end = nodePerGpu
    for i in range(num_gpu):
        gpu_inputs.append(torch.rand((int(end - start), dim)).to("cuda:"+str(i)))
        
        start = end
        end = min(nodePerGpu+end, num_nodes)
    #print(gpu_inputs)
    for t in gpu_inputs:
        new_t = torch.ops.fbgemm.new_managed_tensor(t, t.shape)
        unifi_input.append(new_t)
        zeros_output = torch.zeros_like(t)
        unifi_zero = torch.ops.fbgemm.new_managed_tensor(zeros_output, zeros_output.shape)
        unifi_output.append(zeros_output)
    
    x = torch.rand((50, dim)).to("cuda:1")
    y = x + gpu_inputs[1]
    print(y)
    g = generate_random_digraph(num_nodes, num_edges)
    row_pointers,column_index = getCSR(g)
    
    ps = []
    unifi_output[0][1] = 100
    cpgnn.uvm_test(unifi_output[0], unifi_output)
    for rank in range(num_gpu):
        test(
            unifi_output[rank], 
            unifi_input, 
            row_pointers.cuda(rank),
            column_index.cuda(rank),
            nodePerGpu,
            num_nodes,
            dim,
            num_gpu,
            4,
            rank
            )
        print(unifi_output[rank])
'''
        p = mp.Process(target=test, args=(
            unifi_output[rank], 
            unifi_input, 
            row_pointers,
            column_index,
            nodePerGpu,
            num_nodes,
            dim,
            num_gpu,
            4,
            rank
            ))
        ps.append(p)
    for rank in range(num_gpu):
        print("start rank", rank)
        ps[rank].start()
    for rank in range(num_gpu):
        ps[rank].join()
        print("rank" , rank , "finish")
        print(unifi_output[rank].cpu())
        '''