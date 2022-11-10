CXX=gcc
CXXFLAGS=-fPIC -c -O3 -march=native
LIBFLAGS=--shared

Target = liboptimli.so
BASENAME = liboptimli
SRCS = OptimLi.cpp

library: $(Target)

all: $(Target) test

$(Target): 
	mkdir -p BUILD
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(BASENAME).o
	$(CXX) $(LIBFLAGS) -o $(BASENAME).so $(BASENAME).o
	cp $(BASENAME).so BUILD/

test:
	g++ -fsanitize=address -O0 -lblas -lcblas -llapack Test.cpp

clean:
	rm -rf BUILD
	rm -f a.out
	rm -f $(Target).o
