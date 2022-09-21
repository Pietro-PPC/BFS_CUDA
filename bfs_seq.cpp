#include <iostream>
#include <vector>
#include <stack>
#include <queue>

#define MAX 500000

struct graph_t{
	std::vector< std::vector<int> > gl;
	int size;
};

std::vector<int> bfs(graph_t &g, int src){
	std::vector<int> dist(g.size, -1);

	std::queue<int> q;
	q.push(src);
	dist[src] = 0;
	while (!q.empty()){
		int u = q.front(); q.pop();
		for (auto v : g.gl[u]){
			if (dist[v] == -1){
				dist[v] = dist[u] + 1;
				q.push(v);
			}
		}
	}

	return dist;
}

void component_dfs(graph_t &g, int src, std::vector<bool> &visited){

	std::stack<int> st;
	st.push(src);
	visited[src] = true; 

	while (!st.empty()){
		int u = st.top(); st.pop();
		for (auto v : g.gl[u]){
			if (!visited[v]){
				visited[v] = true;
				st.push(v);
			}
		}
	}

}

int count_components(graph_t &g){
	std::vector<bool> visited(g.size, false);

	int n_components = 0;
	for (int u = 0; u < g.size; ++u)
		if (!visited[u]){
      n_components++;
			component_dfs(g, u, visited);
		}

	return n_components;
}

int main(){
	graph_t g;
  g.gl.resize(MAX);
  g.size = 0;

	int u, v;
	while (std::cin >> u >> v){
		g.gl[u].push_back(v);
		g.gl[v].push_back(u);
		g.size = std::max(g.size, u+1);
		g.size = std::max(g.size, v+1);
	}

	std::vector<int> dist = bfs(g, 0);

	for (int i = 0; i < g.size; ++i){
    std::cout << dist[i] << std::endl;
	}
  
  std::cout << "Número de componentes " << count_components(g) << std::endl;

}
