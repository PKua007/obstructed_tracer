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
#include "random_walker/RandomWalker.h"
#include "move_generator/GaussianMoveGenerator.h"
#include "move_generator/CauchyMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/ImageMoveFilter.h"
#include "image/PPMImageReader.h"

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

    std::ifstream imageFile(parameters.imageFile);
    if (!imageFile)
        die("[main] Cannot open " + parameters.imageFile + " to load image");
    PPMImageReader imageReader;
    Image image = imageReader.read(imageFile);
    std::cout << "[main] Loaded image " << parameters.imageFile << " (" << image.getWidth() << "px x ";
    std::cout << image.getHeight() << "px)" << std::endl;

    std::random_device randomSeed;
    CauchyMoveGenerator moveGenerator(parameters.sigma, randomSeed());
    ImageMoveFilter moveFilter(image, randomSeed());
    RandomWalker randomWalker(parameters.numberOfSteps, parameters.tracerRadius, &moveGenerator, &moveFilter);
    std::cout << "[main] Starting simulation..." << std::endl;
    Trajectory trajectory = randomWalker.run();
    std::cout << "[main] Finished. Initial position: " << trajectory.getFirst() << ", accepted steps: ";
    std::cout << (trajectory.getSize() - 1) << ", final position: " << trajectory.getLast() << std::endl;

    std::string outputFilename = argv[2];
    std::ofstream output(outputFilename);
    if (!output)
        die("[main] Cannot open " + inputFilename + " to store trajectory");

    output << std::fixed << std::setprecision(6);
    trajectory.store(output);
    std::cout << "[main] Trajectory stored to " << outputFilename << std::endl;

    return EXIT_SUCCESS;
}

