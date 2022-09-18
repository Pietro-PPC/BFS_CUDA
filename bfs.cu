#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "chrono.c"
// #include <helper_cuda.h>

// #define LOG true

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
void copy_mem(uint32_t *srcArr, uint32_t *dstArr, long long numElements, int way){
    cudaError_t err = cudaSuccess;

    size_t size = numElements * sizeof(uint32_t);
    err = cudaMemcpy(dstArr, srcArr, size, way);

    if (err != cudaSuccess){
      fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
}

/*
    Aloca vetor com numElements elementos no device
*/
uint32_t *new_device_array(uint32_t numElements){
    cudaError_t err;

    size_t size = numElements * sizeof(uint32_t);
    printf("Alocando vetor de %d elementos\n", numElements);

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
    uint32_t *h_arr = (uint32_t *) malloc(size);

    if (!h_arr ){
        fprintf(stderr, "Falha ao alocar vetor local!\n");
        exit(1);
    }
}

/*
    Desaloca vetor no device
*/
void free_device_array(uint32_t *dev_array){

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
    for (auto u : g){
        vert[vert_i] = list_i;
        for (auto v : g[u]) list[list_i++] = v;

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

    copy_mem(g_vert_dev, g_vert_hos, vert_n, HOS2DEV);
    copy_mem(g_list_dev, g_list_hos, vert_n, HOS2DEV);

    free_host_array(g_vert_hos);
    free_host_array(g_list_hos);

}


/*
 * Host main program
 */
int main(int argc, char *argv[])
{

    std::vector<std::pair<int, int>> edges;
    int vert_n = 0, edge_n;
    while (std::cin >> u >> v){
        edges.push_back({u, v});
        vert_n = max(vert_n, u);
        vert_n = max(vert_n, v);
    }
    vert_n++;
    edge_n = edges.size();

    uint32_t *g_vert_dev, *g_list_dev;
    gen_dev_graph(edges, g_vert_dev, g_list_dev, vert_n);


/*
    long numElements = atol(argv[1]);
    int nblk = atoi(argv[2]);
    char option = argv[3][0];
  
    // Geração de números aleatórios
    int threadsPerBlock = THREADS_PER_BLOCK;
    int usePersistentKernel = 1;

    // Lançamento do kernel persistente CUDA
    curand_kernel_uint32_persT<<<nblk, threadsPerBlock>>>( d_Out1, numElements, SEED );

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch curand_kernel_uint32_persT kernel (error code %s)!\n", 
                         cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    

    // Cálculo do máximo do vetor
    int blocksPerGrid = (numElements/2 + threadsPerBlock - 1) / threadsPerBlock;
    chronometer_t c3;
    
    int curBlocks = blocksPerGrid;
    int curElements = numElements;

    chrono_reset(&c3);
    if (option == 'm'){
      chrono_start(&c3);
      
      while (curBlocks >= 1){
          printf("Chamando kernel de soma para %d elementos com %d bloco%s de %d threads\n", curElements, curBlocks, curBlocks>1 ? "s" : "", threadsPerBlock);
          max_reduction_many<<<curBlocks, threadsPerBlock>>>(d_Out1, curElements);
          
          cudaDeviceSynchronize();

          curElements = curBlocks;
          // Não pode haver número ímpar de elementos
          if (curElements > 1) curElements += curElements%2;

          curBlocks = (curElements/2 + threadsPerBlock - 1) / threadsPerBlock;
      }

      chrono_stop(&c3);
    }
    else if (option == 'p'){
      chrono_start(&c3);
      
      while(curElements > 1){
        printf("Chamando kernel de soma para %d elementos com %d bloco%s de %d threads\n", curElements, nblk, nblk>1 ? "s" : "", threadsPerBlock);
        max_reduction_persistent<<<nblk, threadsPerBlock>>>(d_Out1, curElements);

        cudaDeviceSynchronize();

        // Evitar número ímpar de elementos
        if (curElements > 1) curElements += curElements%2;
        curElements = (curElements/2 + threadsPerBlock - 1) / threadsPerBlock;
      }

      chrono_stop(&c3);
    } else {
      fprintf(stderr, "invalid argument!\n");
      return 1;
    }

    // Copia resultado final para CPU
    uint32_t res;
    copy_from_device(&res , d_Out1, sizeof(uint32_t));

    printf("Máximo: %u\n", res);
    printf("Tempo:  %.3fms\n", chrono_gettotal(&c3)/1000000.0);
    */

    return 0;
}

