# Compiler
MPICXX = mpicxx

# Compilation flags
CXXFLAGS =  -g -I/usr/local/src/AFEPack/library/include -I/usr/local/include/deal.II -L/local/lib
LDFLAGS = -lAFEPack -ldeal_II -lgomp

# Source and output
SRC = compare.cpp
OUT = tr

# Export AFEPACK_TEMPLATE_PATH
export AFEPACK_TEMPLATE_PATH=/usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/include/AFEPack/template/twin_tetrahedron/:/usr/local/include/AFEPack/template/four_tetrahedron/:/usr/local/src/afepack-0.1/library/include/AFEPack/template/four_tetrahedron/ 
# Default target
all: $(OUT)

# Compile
$(OUT): $(SRC)
	$(MPICXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Run the program
run: $(OUT)
	AFEPACK_TEMPLATE_PATH=/usr/local/include/AFEPack/template/triangle/:/usr/local/include/AFEPack/template/twin_triangle/:/usr/local/include/AFEPack/template/tetrahedron/:/usr/local/incude/AFEPack/template/twin_tetrahedron/:/usr/local/inlcude/AFEPack/template/four_tetrahedron/ 
	./$(OUT) D.mesh 

	

# Clean up
clean:
	rm -f $(OUT)
