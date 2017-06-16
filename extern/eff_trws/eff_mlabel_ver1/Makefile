# makefile for programs and generator

CCOMP = g++-4.0
CFLAGS_DEBUG = -g -pg
CFLAGS_RELEASE = -DNDEBUG -pg
CFLAGS = $(CFLAGS_DEBUG)
OBJ_DIR = ./.obj
OUTPUT = ./minimize

all: $(OUTPUT)

$(OUTPUT): $(OBJ_DIR)/main.o $(OBJ_DIR)/image.o
	$(CCOMP) $(CFLAGS) -lm -o $(OUTPUT) $(OBJ_DIR)/main.o $(OBJ_DIR)/image.o

$(OBJ_DIR)/main.o: main.cpp image.h includes.h energy.h graph.h kovtun.h aexpand.h TRWBP.h typePotts.h MRFEnergy.h block.h
	$(CCOMP) $(CFLAGS) -c -o $(OBJ_DIR)/main.o main.cpp

$(OBJ_DIR)/image.o: image.cpp image.h graph.h kovtun.h aexpand.h TRWBP.h block.h energy.h includes.h typePotts.h MRFEnergy.h
	$(CCOMP) $(CFLAGS) -c -o $(OBJ_DIR)/image.o image.cpp

clean:
	rm -f $(OUTPUT)
	rm -f $(OBJ_DIR)/*.o

