#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "chrono.c"

#define LOG true

// #define print_log(X) LOG ? printf(X) : printf("")

#define GPU_NAME "GTX750-Ti"
#define MP 5
#define THREADS_PER_BLOCK 1024
#define THREADS_PER_MP 2048
#define RESIDENT_BLOCKS_PER_MP THREADS_PER_MP/THREADS_PER_BLOCK
#define NTA (MP * RESIDENT_BLOCKS_PER_MP * THREADS_PER_BLOCK)

#define HOS2DEV cudaMemcpyHostToDevice
#define DEV2HOS cudaMemcpyDeviceToHost

/*
    Invólucro para cudaMemcpy para lidar com erros
*/
void copy_mem(uint32_t *dstArr, uint32_t *srcArr, long long numElements, cudaMemcpyKind way){
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(uint32_t);
    err = cudaMemcpy(dstArr, srcArr, size, way);

    if (err != cudaSuccess){
        fprintf(stderr, "Falha ao copiar vetor (Erro: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
    Aloca vetor com numElements elementos no device
*/
uint32_t *new_device_array(uint32_t numElements){
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(uint32_t);
    if (LOG)
        printf("Alocando vetor de %d elementos na GPU\n", numElements); 

    uint32_t *d_arr = NULL;
    err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess){
        fprintf(stderr, "Falha ao alocar vetor (erro %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return d_arr;
}

/*
    Aloca vetor com numElements elementos no host
*/
uint32_t *new_host_array(uint32_t numElements){
    uint32_t *h_arr = (uint32_t *) malloc(numElements * sizeof(uint32_t));

    if (!h_arr ){
        fprintf(stderr, "Falha ao alocar vetor local!\n");
        exit(1);
    }

    return h_arr;
}

/*
    Função auxiliar para imprimir vetor
*/
void print_array(uint32_t *arr, int size){
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

/*
    Desaloca vetor no device
*/
void free_device_array(uint32_t *dev_array){
    cudaError_t err = cudaSuccess;

    err = cudaFree(dev_array);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Falha ao desalocar vetor (erro %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
    Desaloca vetor no host
*/
void free_host_array(uint32_t *hos_array){
    if (hos_array)
        free(hos_array);
}

/*
    Gera lista compacta a partir de lista de adjacências
*/
void gen_compact_list(uint32_t *vert, uint32_t *list, std::vector< std::vector<int> > &g){
    int vert_i = 0, list_i = 0;
    for (auto v_list : g){
        vert[vert_i] = list_i;
        for (auto v : v_list) list[list_i++] = v;

        vert_i++;
    }

    vert[vert_i] = list_i;
}

/*
    Gera grafo no device a partir de vetor de arestas
*/
void gen_dev_graph(std::vector< std::pair<int,int> > &edges, 
        uint32_t **g_vert_dev, uint32_t **g_list_dev, int vert_n){

    std::vector< std::vector<int> > g_hos(vert_n);
    for (auto p : edges){
        g_hos[p.first].push_back(p.second);
        g_hos[p.second].push_back(p.first);
    }

    uint32_t *g_vert_hos, *g_list_hos;
    g_vert_hos = new_host_array( vert_n+1 );
    g_list_hos = new_host_array( edges.size()*2+1 );

    *g_vert_dev = new_device_array( vert_n+1 );
    *g_list_dev = new_device_array( edges.size()*2+1 );

    gen_compact_list(g_vert_hos, g_list_hos, g_hos);

    copy_mem(*g_vert_dev, g_vert_hos, vert_n+1, HOS2DEV);
    copy_mem(*g_list_dev, g_list_hos, edges.size()*2+1, HOS2DEV);

    free_host_array(g_vert_hos);
    free_host_array(g_list_hos);
}


__global__ 
bool process_frontier(
    uint32_t *g_vert, uint32_t *g_list, 
    uint32_t *dist, uint32_t *proc, uint32_t *fron, 
    int vert_sz, int list_sz, uint32_t *ended){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int vertIdx = bid*bdim + tid;


    if (vertIdx < vert_sz-1 && fron[vertIdx]){
        fron[vertIdx] = 0;
        proc[vertIdx] = 1;
        for (int i = g_vert[vertIdx]; i < g_vert[vertIdx+1]; ++i){
            if (!proc[ g_list[i] ]){
                fron[ g_list[i] ] = 1;
                dist[ g_list[i] ] = dist[vertIdx] + 1;
            }
        }
    }

    __syncthreads();
    if ( fron[vertIdx] ) *ended = 1;

}

/*
 * Host main program
 */
int main(int argc, char *argv[])
{
    std::vector<std::pair<int, int>> edges;
    int vert_n = 0, edge_n;
    int u, v;
    while (std::cin >> u >> v){
        edges.push_back({u, v});
        vert_n = std::max(vert_n, u);
        vert_n = std::max(vert_n, v);
    }
    vert_n++;
    edge_n = edges.size();

    uint32_t *g_vert_dev, *g_list_dev;
    gen_dev_graph(edges, &g_vert_dev, &g_list_dev, vert_n);

    uint32_t *g_vert_hos = new_host_array(vert_n+1);



    uint32_t *dist = new_device_array(vert_n);
    uint32_t *proc = new_device_array(vert_n);
    uint32_t *fron = new_device_array(vert_n);

    int n_blocks = (vert_n + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    __device__ uint32_t ended_dev = false;


    process_frontier<<<n_blocks, THREADS_PER_BLOCK>>>(
        g_vert_dev, g_list_dev, 
        dist, proc, fron, 
        vert_n+1, edges.size()*2 + 1, &ended_dev);
    
    uint32_t ended_hos;
    copy_mem(&ended_hos, &ended_dev, 1, DEV2HOS);


    uint32_t *hos_fron = new_host_array(vert_n);
    copy_mem(hos_fron, fron, vert_n, DEV2HOS);
    print_array(hos_fron, vert_n);



    free_device_array(g_vert_dev);
    free_device_array(g_list_dev);
    free_device_array(dist);
    free_device_array(proc);
    free_device_array(fron);

    return 0;
}

