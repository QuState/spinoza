CC  = gcc
CXX = g++

QULACS_PATH?=$(HOME)/repos/qulacs

# Additional include directories
INCLUDES = $(QULACS_PATH)/include

# Compilation options:
CXXFLAGS = -Wall -Wextra -O3 -march=native -I $(INCLUDES)

# Linking options:
LDFLAGS = -L $(QULACS_PATH)/lib

# Libraries needed to link with
LDLIBS = -lvqcsim_static -lcppsim_static -lcsim_static -fopenmp


rz: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_rz.cpp -o qulacs_benchmark_rz $(LDLIBS) $(LDFLAGS)

rx: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_rx.cpp -o qulacs_benchmark_rx $(LDLIBS) $(LDFLAGS)

ry: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_ry.cpp -o qulacs_benchmark_ry $(LDLIBS) $(LDFLAGS)

x: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_x.cpp -o qulacs_benchmark_x $(LDLIBS) $(LDFLAGS)

h: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_h.cpp -o qulacs_benchmark_h $(LDLIBS) $(LDFLAGS)

p: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_p.cpp -o qulacs_benchmark_p $(LDLIBS) $(LDFLAGS)

z: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_z.cpp -o qulacs_benchmark_z $(LDLIBS) $(LDFLAGS)

qcbm: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_qcbm.cpp -o qulacs_benchmark_qcbm $(LDLIBS) $(LDFLAGS)

value_encoding: utils.h
	$(CXX) $(CXXFLAGS) qulacs_benchmark_value_encoding.cpp -o qulacs_benchmark_value_encoding $(LDLIBS) $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) qulacs_benchmark_{rz,rx,ry,x,h,p,z,qcbm,value_encoding}

.PHONY: all
all: clean rz rx ry x h p z qcbm value_encoding
