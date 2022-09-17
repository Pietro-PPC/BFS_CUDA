#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "chrono.c"
// #include <helper_cuda.h>



#define GPU_NAME "GTX750-Ti"
#define MP 5
#define THREADS_PER_BLOCK 1024   // can be defined in the compilation line with -D
#define THREADS_PER_MP 2048
#define RESIDENT_BLOCKS_PER_MP THREADS_PER_MP/THREADS_PER_BLOCK
#define NTA \
  (MP * RESIDENT_BLOCKS_PER_MP * THREADS_PER_BLOCK)


// void copy_from_device(uint32_t *hostArr, uint32_t *devArr, long size){
//   //printf("Copy output data from the CUDA device to the host memory\n");

//   cudaError_t err = cudaSuccess;
//   err = cudaMemcpy(hostArr, devArr, size, cudaMemcpyDeviceToHost);

//   if (err != cudaSuccess)
//   {
//       fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
//       exit(EXIT_FAILURE);
//   }
// }

/*
 * Host main program
 */
int main(int argc, char *argv[])
{

    std::vector<std::pair<int, int>> edges;

    int u, v, max_vert = 0;
    while (std::cin >> u >> v){
      edges.push_back({u, v});
      max_vert = max(max_vert, u);
      max_vert = max(max_vert, v);
    }
    max_vert++;

    thrust::device_vector< uint32_t > g_dev(max_vert, uint32_t);
    // thrust::host_vector< thrust::host_vector<int> > g_host(max_vert, thrust::device_vector<int>);

/*
    long numElements = atol(argv[1]);
    int nblk = atoi(argv[2]);
    char option = argv[3][0];

    // Alocações
    size_t size = numElements * sizeof(uint32_t);  // alloca a big vector all generated random numbers
    printf("[Generate %ld random numbers, output vector size %ld Bytes]\n",
               numElements, size); //(1024*1024));
    

    uint32_t *d_Out1 = NULL;
    err = cudaMalloc((void **)&d_Out1, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    uint32_t *h_Out1 = (uint32_t *) malloc(size);
    if (!h_Out1 ){
      fprintf(stderr, "Error allocating local array!\n");
      exit(1);
    }

  
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


    // DEALLOC MEMORY
    err = cudaFree(d_Out1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_Out1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }*/

    return 0;
}

