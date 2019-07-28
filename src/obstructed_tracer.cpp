/*
 ============================================================================
 Name        : obstructed_tracer.cpp
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
#include "RandomWalker.h"

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

    RandomWalker randomWalker(0.f, 0.f, parameters.sigma, parameters.numberOfSteps);
    std::cout << "[main] Starting simulation..." << std::endl;
    Trajectory trajectory = randomWalker.run();
    std::cout << "[main] Finished. Trajectory size: " << trajectory.size() << ", final position: ";
    std::cout << trajectory.back() << std::endl;

    return EXIT_SUCCESS;
}

