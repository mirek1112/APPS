# Název výsledného binárního souboru (název složky)
MAIN_NAME := $(notdir $(shell pwd))

# Kompilační a linkovací přepínače
CPPFLAGS  += -g -I/usr/local/cuda/include
LDFLAGS   += -L/usr/local/cuda/lib64
LDLIBS    = -lcudart -lcuda

# Správná CUDA architektura pro GTX 1060
NVCCFLAGS += -gencode arch=compute_61,code=sm_61

# Automatická detekce OpenCV přes pkgconf
OPENCVLIB := $(shell pkgconf --list-package-names | grep opencv)

ifeq ($(OPENCVLIB),)

all:
	@echo "❌ OpenCV knihovna nebyla nalezena!"
	@exit 1

else

# OpenCV příznaky pro C++ kompilaci
CPPFLAGS      += $(shell pkgconf --cflags $(OPENCVLIB))
# Jen include path pro nvcc (aby nechytil -std=c++xx)
OPENCV_CFLAGS := $(shell pkgconf --cflags-only-I $(OPENCVLIB))
LDFLAGS       += $(shell pkgconf --libs-only-L $(OPENCVLIB))
LDLIBS        += $(shell pkgconf --libs-only-l $(OPENCVLIB)) -lcudart -lcuda

# Soubory
CPP_SRCS  := $(wildcard *.cpp)
CU_SRCS   := $(wildcard *.cu)
HEADERS   := $(wildcard *.h)

CPP_OBJS  := $(CPP_SRCS:.cpp=.o)
CU_OBJS   := $(CU_SRCS:.cu=.o)

all: $(MAIN_NAME)

endif

# Kompilace C++ souborů
%.o: %.cpp $(HEADERS)
	g++ -c $(CPPFLAGS) $< -o $@

# Kompilace CUDA souborů (pouze include path z OpenCV)
%.o: %.cu $(HEADERS)
	nvcc -c -std=c++11 $(NVCCFLAGS) $(OPENCV_CFLAGS) $< -o $@

# Linkování finálního programu
$(MAIN_NAME): $(CPP_OBJS) $(CU_OBJS)
	g++ $(CPPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Úklid
clean:
	rm -f *.o $(MAIN_NAME)
