PROG=track
OPTS= -O3 -std=c++14
ARUCO=-laruco
include ../Makefile.opencv

all: $(PROG)

$(PROG): main.cpp
	g++ $(OPTS) $< -o $@ $(ARUCO) $(OPENCV)

clean:
	- rm $(PROG)

