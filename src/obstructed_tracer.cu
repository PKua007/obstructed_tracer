/*
 ============================================================================
 Name        : obstructed_tracer.cu
 Author      : Piotr Kubala
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <cstdlib>
#include <fstream>

#include "Utils.h"
#include "Parameters.h"

int main(int argc, char **argv)
{
    std::string command = argv[0];
    if (argc < 2)
        die("[main] Usage: " + command + " [input file]");

    std::string inputFilename = argv[1];
    std::ifstream input(inputFilename);
    if (!input)
        die("[main] Cannot open " + inputFilename + " to read parameters");

    Parameters parameters(input);

    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    parameters.print(std::cout);

    return EXIT_SUCCESS;
}

