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
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <random>

#include "Parameters.h"
#include "utils/Utils.h"
#include "SimulationFactory.h"

int main(int argc, char **argv)
{
    std::string command = argv[0];
    if (argc < 3)
        die("[main] Usage: " + command + " [input file] [output file]");

    std::string inputFilename = argv[1];
    std::ifstream input(inputFilename);
    if (!input)
        die("[main] Cannot open " + inputFilename + " to read parameters");

    Parameters parameters(input);
    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    parameters.print(std::cout);
    std::cout << std::endl;

    SimulationFactory simulationFactory(parameters, std::cout);
    RandomWalker &randomWalker = simulationFactory.getRandomWalker();
    randomWalker.run(std::cout);

    /*std::cout << "[main] Starting simulation " << i << "... " << std::flush;
    Trajectory trajectory = simulationFactory.getRandomWalker().run();
    std::cout << "Finished. Initial position: " << trajectory.getFirst() << ", accepted steps: ";
    std::cout << trajectory.getNumberOfAcceptedSteps() << ", final position: " << trajectory.getLast() << std::endl;*/

    /*std::string outputFilename = argv[2];
    std::ofstream output(outputFilename);
    if (!output)
        die("[main] Cannot open " + inputFilename + " to store trajectory");

    output << std::fixed << std::setprecision(6);
    trajectory.store(output);
    std::cout << "[main] Trajectory stored to " << outputFilename << std::endl;*/

    return EXIT_SUCCESS;
}

