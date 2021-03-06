
BIN_DIR=./bin
LIB_DIR=./lib

# BINS=$(BIN_DIR)/output_test
BINS=$(BIN_DIR)/output_test $(BIN_DIR)/output_test_with_config
# BINS=$(BIN_DIR)/output_test $(BIN_DIR)/output_stencil
# BINS=$(BIN_DIR)/output_stencil $(BIN_DIR)/output_dot_product $(BIN_DIR)/output_matrix_mul

# GECKO_FILES=geckoGraph.cpp geckoRuntime.cpp geckoUtilsAcc.cpp 
# GECKO_FILES=geckoRuntime.cpp geckoHierarchicalTree.cpp geckoDataTypeGenerator.cpp
GECKO_FILES=$(wildcard gecko*.cpp)
GECKO_OBJ_FILES=$(GECKO_FILES:.cpp=.o)
GECKO_LIB_FILE=$(LIB_DIR)/libgecko.a

CUDA_HOME?=/usr/local/cuda

ENABLE_CUDA = ON
ENABLE_DEBUG = OFF


ENABLE_TCMALLOC = OFF
TC_MALLOC_LOCATION = /usr/lib/x86_64-linux-gnu


ENABLE_JEMALLOC = OFF
JE_MALLOC_LOCATION = /usr/lib/x86_64-linux-gnu


LDFLAGS=-lm

# NVCC=nvcc
# CUDAFLAGS=-w -std c++11


# CXX=g++
# CXXFLAGS=-w -fopenmp -fopenacc -lcuda
# LDFLAGS=-lm
# OUTPUT_EXE=output_test
# CXXFLAGS += -acc -ta=tesla,multicore


CXX=pgc++
CXXFLAGS=-m64 -std=c++11 -w
#CXXFLAGS= -std=c++11 -w
CXXFLAGS+=-Mllvm 
CXXFLAGS+=-mp
LDFLAGS=-lm
OUTPUT_EXE=output_test
CXXFLAGS += -acc -ta=tesla,multicore -Minfo=accel

ifeq ($(ENABLE_TCMALLOC), ON)
	LDFLAGS += $(TC_MALLOC_LOCATION)/libtcmalloc.so.4
endif

ifeq ($(ENABLE_JEMALLOC), ON)
	LDFLAGS += -L$(JE_MALLOC_LOCATION) -ljemalloc
endif

CXXFLAGS_DEBUG=-O0 -g 
CXXFLAGS_RELEASE=-O3 


ifeq ($(ENABLE_DEBUG), ON)
CXXFLAGS += -DINFO
endif


OUTPUT_EXE=output_test 
STENCIL_EXE = output_stencil
DOT_PRODUCT_EXE = output_dot_product
MATRIX_MUL_EXE = output_matrix_mul
OUTPUT_EXE_WITH_CONF = output_test_with_config





ifeq ($(ENABLE_CUDA), ON)
CXXFLAGS += -DCUDA_ENABLED -I$(CUDA_HOME)/include/
CUDAFLAGS+= -DCUDA_ENABLED -I$(CUDA_HOME)/include/
LDFLAGS += 	-L$(CUDA_HOME)/lib64 -lcudart 
endif



ifeq ($(ENABLE_DEBUG), ON)
CXXFLAGS+=$(CXXFLAGS_DEBUG) -DDEBUG
CUDAFLAGS+=-DDEBUG -O0 -g
else
CXXFLAGS+=$(CXXFLAGS_RELEASE)
endif



.PHONY: doTransformation clean



all: doTransformation lib $(BINS)
# all: lib $(BINS)


$(BIN_DIR):
	mkdir -p $(BIN_DIR)

doTransformation:
	python geckoTranslate.py


$(BIN_DIR)/test: test.o $(BIN_DIR)
	${CXX} ${CXXFLAGS} test.cpp -o $(BIN_DIR)/test


lib: $(GECKO_OBJ_FILES) 
	mkdir -p $(LIB_DIR)
	ar rc $(GECKO_LIB_FILE) $(GECKO_OBJ_FILES)
	ranlib $(GECKO_LIB_FILE)

# %.o: %.cpp
# 	${NVCC} ${CXXFLAGS} -c $< -o $@

# geckoHierarchicalTree.o: geckoHierarchicalTree.cpp 
# 	$(CXX) $(CXXFLAGS) -c $< -o $@
# 	# $(NVCC) $(CUDAFLAGS) -c $< -o $@

# geckoRuntime.o: geckoRuntime.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@
# 	# $(NVCC) $(CUDAFLAGS) -c $< -o $@

# # geckoUtilsAcc.o: geckoUtilsAcc.cpp
# # 	$(CXX) $(CXXFLAGS) -c $< -o $@

# geckoDataTypeGenerator.o: geckoDataTypeGenerator.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@






$(BIN_DIR)/${OUTPUT_EXE}: ${GECKO_LIB_FILE} $(BIN_DIR) output_test.cpp 
	${CXX} ${CXXFLAGS} -o $(BIN_DIR)/${OUTPUT_EXE} output_test.cpp ${LDFLAGS} ${GECKO_LIB_FILE} 

$(BIN_DIR)/${OUTPUT_EXE_WITH_CONF}: ${GECKO_OBJ_FILES} $(BIN_DIR) output_test_with_config.cpp 
	${CXX} ${CXXFLAGS} -o $(BIN_DIR)/${OUTPUT_EXE_WITH_CONF} ${GECKO_OBJ_FILES} output_test_with_config.cpp ${LDFLAGS}

$(BIN_DIR)/${STENCIL_EXE}: ${GECKO_OBJ_FILES} $(BIN_DIR) output_stencil.cpp
	${CXX} ${CXXFLAGS} -o $(BIN_DIR)/${STENCIL_EXE} ${GECKO_OBJ_FILES} output_stencil.cpp ${LDFLAGS}

$(BIN_DIR)/${DOT_PRODUCT_EXE}: ${GECKO_OBJ_FILES} $(BIN_DIR) output_dot_product.cpp
	${CXX} ${CXXFLAGS} -o $(BIN_DIR)/${DOT_PRODUCT_EXE} ${GECKO_OBJ_FILES} output_dot_product.cpp ${LDFLAGS}

$(BIN_DIR)/${MATRIX_MUL_EXE}: ${GECKO_OBJ_FILES} $(BIN_DIR) output_matrix_mul.cpp
	${CXX} ${CXXFLAGS} -o $(BIN_DIR)/${MATRIX_MUL_EXE} ${GECKO_OBJ_FILES} output_matrix_mul.cpp ${LDFLAGS}

dot: $(wildcard *.dot)
	dot -Tpdf $< -o $<.pdf

clean:
	rm -rf *.o $(BIN_DIR) $(LIB_DIR) *.dot

