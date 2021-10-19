CFLAGS=-I.
OPTS=-g -O4
include ../Makefile.opencv

all: realsense

realsense: realsense.cpp
	g++ $(OPTS) -std=c++14 $(CFLAGS) $< -o $@ -lrealsense2 $(OPENCV)

clean:
	- rm realsense
