CXX=gcc
CXXFLAGS=-fPIC -c -O3 -march=native
LIBFLAGS=--shared

Target=OptimLi

$(Target): $(Target).cpp
	mkdir BUILD
	$(CXX) $(CXXFLAGS) $(Target).cpp -o $(Target).o
	$(CXX) $(LIBFLAGS) -o lib$(Target).so $(Target).o
	mv lib$(Target).so BUILD/

test:
	g++ -fsanitize=address -O0 -lblas -lcblas -llapack Test.cpp

clean:
	rm -rf BUILD
	rm -f a.out
	rm -f $(Target).o
