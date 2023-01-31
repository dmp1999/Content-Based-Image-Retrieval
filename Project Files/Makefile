# Project 1 of CS 5330: Pattern Recognition and Computer Vision
# Created by Dhruvil Parikh

CC = "g++"

SRC = main.cpp
PROJECT = main

LIBS = `pkg-config opencv4 --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)