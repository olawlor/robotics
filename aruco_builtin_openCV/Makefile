PROG=track
OPTS=-O
include ../Makefile.opencv
ARUCO=-lopencv_aruco


all: $(PROG)

$(PROG): main.cpp
	g++ $(OPTS) $< -o $@ $(ARUCO) $(OPENCV)

draw_marker: draw_marker.cpp
	g++ $(OPTS) $< -o $@ $(ARUCO) $(OPENCV)

run: $(PROG)
	./$(PROG)

clean:
	- rm $(PROG)

