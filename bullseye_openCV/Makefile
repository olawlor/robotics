PROG=track
OPTS=-O
include ../Makefile.opencv


all: $(PROG)

$(PROG): bullseye.cpp
	g++ $(OPTS) $< -o $@ $(OPENCV)

run: $(PROG)
	./$(PROG)

clean:
	- rm $(PROG)

