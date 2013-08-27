PROG=track
OPTS=-O
OPENCV=-I /usr/local/include/opencv -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_features2d -lopencv_ts -lopencv_imgproc -lopencv_video -lm 

all: $(PROG)

$(PROG): colortrack.cpp
	g++ $(OPTS) $< -o $@ $(OPENCV)

run: $(PROG)
	./$(PROG)

clean:
	- rm $(PROG)

