PROG=track
OPTS= -O3 -std=c++14
OPENCV=`pkg-config opencv --cflags --libs`
ARUCO=-laruco


all: $(PROG)

$(PROG): main.cpp
	g++ $(OPTS) $< -o $@ $(ARUCO) $(OPENCV)

clean:
	- rm $(PROG)

