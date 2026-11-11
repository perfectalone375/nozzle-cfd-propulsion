CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
CXXFLAGS20 ?= -O3 -std=c++20 -Wall -Wextra -Wpedantic
CXXOPENMP ?= -fopenmp

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Linux)
	LIB_EXT = libnozzle_fvm_core.so
	LIB2D_EXT = libnozzle_fvm2d_core.so
	LIB_FLAGS = -shared -fPIC
else
	LIB_EXT = nozzle_fvm_core.dll
	LIB2D_EXT = nozzle_fvm2d_core.dll
	LIB_FLAGS = -shared -static
endif

all: maccormack_cfd hllc_fvm_cfd axisymmetric_ns2d $(LIB_EXT) $(LIB2D_EXT)

maccormack_cfd: maccormack_cfd.cpp
	$(CXX) $(CXXFLAGS) -o maccormack_cfd maccormack_cfd.cpp

hllc_fvm_cfd: hllc_fvm_solver.cpp
	$(CXX) $(CXXFLAGS20) -o hllc_fvm_cfd hllc_fvm_solver.cpp

$(LIB_EXT): hllc_fvm_solver.cpp
	$(CXX) $(CXXFLAGS20) $(LIB_FLAGS) -DNOZZLE_FVM_BUILD_DLL -o $(LIB_EXT) hllc_fvm_solver.cpp

axisymmetric_ns2d: axisymmetric_ns2d_solver.cpp
	$(CXX) $(CXXFLAGS20) $(CXXOPENMP) -o axisymmetric_ns2d axisymmetric_ns2d_solver.cpp -lm

$(LIB2D_EXT): axisymmetric_ns2d_solver.cpp
	$(CXX) $(CXXFLAGS20) $(CXXOPENMP) $(LIB_FLAGS) -DNOZZLE_FVM2D_BUILD_DLL -o $(LIB2D_EXT) axisymmetric_ns2d_solver.cpp -lm

test: all
	./maccormack_cfd --pc-mpa 10.0 --pa-kpa 101.325 --eps 12.0 --json
	./maccormack_cfd --pc-mpa 1.0 --pa-kpa 380.0 --eps 8.0 --json
	./hllc_fvm_cfd --nx 161 --pc-mpa 10.0 --pa-kpa 101.325 --eps 12.0 --flux hllc --limiter van_leer --json
	./hllc_fvm_cfd --nx 161 --pc-mpa 1.0 --pa-kpa 380.0 --eps 8.0 --flux hllc --limiter van_leer --json
	./axisymmetric_ns2d --json

clean:
	rm -f maccormack_cfd hllc_fvm_cfd axisymmetric_ns2d *.exe *.dll *.so
