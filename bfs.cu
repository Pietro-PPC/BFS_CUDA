#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "chrono.c"

#define LOG false

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
    Função auxiliar para imprimir vetor
*/
void print_array(uint32_t *arr, int size){
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

/************* INVÓLUCROS DE FUNÇÕES CUDA ************/

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


/********** FUNÇÕES DE ALOCAÇÃO ***********/

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

/************** GERAÇÃO DE GRAFO *****************/

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
        uint32_t *g_vert_dev, uint32_t *g_list_dev, int vert_n, int edge_n){
    // Aloca vetor de vértices e lista de adjacências no host
    uint32_t *g_vert_hos, *g_list_hos;
    g_vert_hos = new_host_array( vert_n+1 );
    g_list_hos = new_host_array( edge_n*2+1 );

    // Gera lista de adjacências no host
    std::vector< std::vector<int> > g_hos(vert_n);
    for (auto p : edges){
        g_hos[p.first].push_back(p.second);
        g_hos[p.second].push_back(p.first);
    }

    // Gera lista compacta de adjacências no host
    gen_compact_list(g_vert_hos, g_list_hos, g_hos);

    copy_mem(g_vert_dev, g_vert_hos, vert_n+1, HOS2DEV);
    copy_mem(g_list_dev, g_list_hos, edge_n*2+1, HOS2DEV);

    free_host_array(g_vert_hos);
    free_host_array(g_list_hos);
}


/************* CÁLCULO DA BFS ************/

/*
    Kernel que processa todos os vértices da fronteira e atualiza os próximos
    a serem processados
        . g_vert: vetor de apontadores para início das listas de adjacências de cada vértice
        . g_list: vetor com todas as listas de adjacências
        . dist: vetor de distâncias dos vértices
        . proc: vetor de vértices processados
        . fron: vetor de vértices na fronteira
        . vert_sz e list_sz: números de elementos de g_vert e g_list, respectivamente
        . ended: informa se não há mais vértices a serem processados 
*/
__global__ 
void advance_frontier(
    uint32_t *g_vert, uint32_t *g_list, 
    uint32_t *dist, uint32_t *proc, uint32_t *fron, 
    int vert_sz, int list_sz, uint32_t *ended){

    // Calcula índice da thread no total.
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int vertIdx = bid*bdim + tid;

    // Garante que thread tem indice válido
    if (vertIdx >= vert_sz-1) return;

    // Processa vértices na fronteira
    if (fron[vertIdx]){
        fron[vertIdx] = 0;
        proc[vertIdx] = 1;
        for (int i = g_vert[vertIdx]; i < g_vert[vertIdx+1]; ++i){
            if (!proc[ g_list[i] ]){
                fron[ g_list[i] ] = 1;
                dist[ g_list[i] ] = dist[vertIdx] + 1;
            }
        }
    }

    // inicializa ended como 1
    if (vertIdx == 0) *ended = 1;
    __syncthreads();

    // Se fronteira não estiver vazia, ended = 0
    if ( fron[vertIdx] ) *ended = 0;

}

/*
    Inicializa valores iniciais dos vetores dist, proc e fron
        . dist_dev corresponde às distâncias calculadas pela BFS
        . proc_dev informa se cada vértice foi processado ou não
        . fron_dev informa se cada vértice será processado na próxima iteração ou não
*/
void initialize_aux_arrays(uint32_t *dist_dev, uint32_t *proc_dev, uint32_t *fron_dev, int vert_n){
    cudaMemset(dist_dev, 0xff, vert_n*sizeof(uint32_t)); // Distância começa com "infinito"
    cudaMemset(proc_dev, 0, vert_n*sizeof(uint32_t));    // Nenhum vértice foi processado
    cudaMemset(fron_dev, 0, vert_n*sizeof(uint32_t));    // Nenhum vértice está na fronteira

    int val;
    // Primeiro vértice tem distância 0 e pertence à fronteira.
    val = 0; cudaMemcpy(dist_dev, &val, sizeof(uint32_t), HOS2DEV); // dist[0] = 0
    val = 1; cudaMemcpy(fron_dev, &val, sizeof(uint32_t), HOS2DEV); // fron[0] = 1
}


/*
    Roda BFS calculando vetor de distâncias
*/
void calculate_bfs(uint32_t *dist_hos, uint32_t *g_vert_dev, uint32_t *g_list_dev, int vert_n, int edge_n){

    // Aloca e inicializa vetores de distância, vértices processados e fronteira na GPU
    uint32_t *dist_dev = new_device_array(vert_n);
    uint32_t *proc_dev = new_device_array(vert_n);
    uint32_t *fron_dev = new_device_array(vert_n);
    initialize_aux_arrays(dist_dev, proc_dev, fron_dev, vert_n);

    // Inicializa variáveis auxiliares
    int n_blocks = (vert_n + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    uint32_t *ended_dev = new_device_array(1);
    uint32_t ended_hos = 0;
    
    // Processa vértices até não haver ninguém na fronteira
    if (LOG) printf("%d blocos de %d threads.\n", n_blocks, THREADS_PER_BLOCK);
    while (!ended_hos){
        advance_frontier<<<n_blocks, THREADS_PER_BLOCK>>>(
            g_vert_dev, g_list_dev, 
            dist_dev, proc_dev, fron_dev, 
            vert_n+1, edge_n*2 + 1, ended_dev);
        cudaDeviceSynchronize();
        
        copy_mem(&ended_hos, ended_dev, 1, DEV2HOS); // Copia variável 
    }
    copy_mem(dist_hos, dist_dev, vert_n, DEV2HOS);

    free_device_array(dist_dev);
    free_device_array(proc_dev);
    free_device_array(fron_dev);
}

/*
 * Host main program
 */
int main(int argc, char *argv[])
{
    std::vector<std::pair<int, int>> edges;
    int vert_n = 0, edge_n;
    int u, v;

    // Leitura de entrada
    while (std::cin >> u >> v){
        edges.push_back({u, v});
        vert_n = std::max(vert_n, u);
        vert_n = std::max(vert_n, v);
    }
    vert_n++;
    edge_n = edges.size();

    // Aloca e inicializa grafo na GPU a partir do vetor de arestas
    uint32_t *g_vert_dev = new_device_array( vert_n+1 );
    uint32_t *g_list_dev = new_device_array( edge_n*2+1 );
    gen_dev_graph(edges, g_vert_dev, g_list_dev, vert_n, edge_n);
    
    // Aloca vetor de distâncias roda BFS 
    uint32_t *dist_hos = new_host_array(vert_n);
    calculate_bfs(dist_hos, g_vert_dev, g_list_dev, vert_n, edge_n);

    // Libera memória
    free_host_array(dist_hos);
    free_device_array(g_vert_dev);
    free_device_array(g_list_dev);

    return 0;
}

