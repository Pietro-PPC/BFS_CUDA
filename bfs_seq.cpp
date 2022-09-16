#include <iostream>
#include <vector>
#include <stack>

#define MAX 5000

struct graph_t{
	std::vector<int> gl[MAX];
	int size;
};


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

int count_components(graph_t g){
	std::vector<bool> visited(g.size, false);

	int n_components = 0;
	for (int u = 0; u < g.size; ++u)
		if (!visited[u]){
			std::cout << n_components++ << std::endl;
			component_dfs(g, u, visited);
		}

	return n_components;
}

int main(){
	graph_t g;
	std::vector<int> dist[MAX];

	int u, v;
	while (std::cin >> u >> v){
		g.gl[u].push_back(v);
		g.gl[v].push_back(u);
		g.size = std::max(g.size, u+1);
		g.size = std::max(g.size, v+1);
	}

	int comps = count_components(g);
	std::cout << comps << std::endl;

}
