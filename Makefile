all: bfs_seq.out

bfs_seq.out: bfs_seq.cpp
	g++ bfs_seq.cpp -o bfs_seq.out

purge: 
	rm *.out


