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

#include "Parameters.h"
#include "MSDData.h"
#include "Simulation.h"
#include "utils/Utils.h"

#include "simulation/SimulationImpl.h"


int main(int argc, char **argv){
    std::string command = argv[0];
    if (argc < 3)
        die("[main] Usage: " + command + " [input file] [output file prefix]");

    std::string inputFilename = argv[1];
    std::ifstream inputFile(inputFilename);
    if (!inputFile)
        die("[main] Cannot open " + inputFilename + " to read parameters");

    Parameters parameters(inputFile);
    std::cout << "[main] Parameters loaded from " + inputFilename << ":" << std::endl;
    parameters.print(std::cout);
    std::cout << std::endl;

    std::string outputFilePrefix = argv[2];

    auto simulation = std::unique_ptr<Simulation>(new SimulationImpl(parameters, outputFilePrefix, std::cout));
    simulation->run(std::cout);
    MSDData &msdData = simulation->getMSDData();

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        die("[main] Cannot open " + msdFilename + " to store mean square displacement data");
    msdData.store(msdFile);
    std::cout << "[main] Mean square displacement data stored to " + msdFilename << std::endl;

    std::cout << "[main] Run finished." << std::endl;
    return EXIT_SUCCESS;
}

