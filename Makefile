INCPATH=-I/usr/include/opencv -I/usr/include/pcl-1.7 -I/usr/include/eigen3 -I/usr/include/vtk-6.2 -I/home/bence/Documents/Workspace/vpcap/inc -I/home/bence/Documents/Workspace/TauronExample
LIBPATH=-L/usr/lib/x86_64-linux-gnu/ -L/home/bence/Documents/Workspace/vpcap/lib
LIBS=-lvpcap -lopencv_core -lopencv_highgui -lopencv_imgproc -lpcl_common -lpcl_visualization -lpcl_io -lboost_system
CXXFLAGS=-Wall -pedantic -std=c++14
CXX=g++

all: main.cpp
	$(CXX) main.cpp $(CXXFLAGS) $(INCPATH) $(LIBPATH) $(LIBS) -o main 

clean:
	rm main

run:
	./main