CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

all: maccormack_cfd

maccormack_cfd: maccormack_cfd.cpp
	$(CXX) $(CXXFLAGS) -o maccormack_cfd maccormack_cfd.cpp

test: maccormack_cfd
	./maccormack_cfd --pc-mpa 10.0 --pa-kpa 101.325 --eps 12.0 --json
	./maccormack_cfd --pc-mpa 1.0 --pa-kpa 380.0 --eps 8.0 --json

clean:
	rm -f maccormack_cfd
