CC= g++
CFLAGS= -O2 -Wall -g

all: bfs_seq

bfs_seq: bfs_seq.cpp
	$(CC) bfs_seq.cpp -o bfs_seq.out $(CFLAGS)

purge: 
	rm *.out
