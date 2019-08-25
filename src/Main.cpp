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
#include <iterator>

#include "Parameters.h"
#include "utils/Utils.h"
#include "SimulationFactory.h"

namespace {
    struct MSDData {
        float x2{};
        float y2{};
        float xy{};
    };

    std::ostream &operator<<(std::ostream &out, MSDData msdData) {
        out << msdData.x2 << " " << msdData.y2 << " " << msdData.xy;
    }
}

int main(int argc, char **argv)
{
    std::string command = argv[0];
    if (argc < 3)
        die("[main] Usage: " + command + " [input file] [output file prefix]");

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

    std::string outputFilePrefix = argv[2];
    std::size_t numberOfTrajectories = randomWalker.getNumberOfTrajectories();
    std::vector<MSDData> msdDatas(randomWalker.getTrajectory(0).getSize());
    for (std::size_t i = 0; i < numberOfTrajectories; i++) {
        auto &trajectory = randomWalker.getTrajectory(i);

        float startX = trajectory[0].x;
        float startY = trajectory[0].y;
        for (std::size_t j = 0; j < trajectory.getSize(); j++) {
            float x = trajectory[j].x - startX;
            float y = trajectory[j].y - startY;
            msdDatas[j].x2 += x*x;
            msdDatas[j].y2 += y*y;
            msdDatas[j].xy += x*y;
        }
    }

    for (auto &msdData : msdDatas) {
        msdData.x2 /= numberOfTrajectories;
        msdData.y2 /= numberOfTrajectories;
        msdData.xy /= numberOfTrajectories;
    }

    std::string msdFilename = outputFilePrefix + "_msd.txt";
    std::ofstream msdFile(msdFilename);
    if (!msdFile)
        die("[main] Cannot open " + msdFilename + " to store mean square displacement data");
    std::copy(msdDatas.begin(), msdDatas.end(), std::ostream_iterator<MSDData>(msdFile, "\n"));
    std::cout << "[main] Mean square displacement data stored to " + msdFilename << std::endl;

    /*std::ofstream output(outputFilename);
    if (!output)
        die("[main] Cannot open " + inputFilename + " to store trajectory");

    output << std::fixed << std::setprecision(6);
    trajectory.store(output);
    std::cout << "[main] Trajectory stored to " << outputFilename << std::endl;*/

    return EXIT_SUCCESS;
}

