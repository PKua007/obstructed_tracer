/*
 * SimulationFactory.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>

#include "SimulationFactory.h"
#include "move_generator/GaussianMoveGenerator.h"
#include "move_generator/CauchyMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/image_move_filter/ImageMoveFilter.h"
#include "move_filter/image_move_filter/WallBoundaryConditions.h"
#include "move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"

SimulationFactory::SimulationFactory(Parameters parameters, std::ostream &logger) {
    std::ifstream imageFile(parameters.imageFile);
    if (!imageFile)
        throw std::runtime_error("Cannot open " + parameters.imageFile + " to load image");
    PPMImageReader imageReader;
    Image image = imageReader.read(imageFile);
    logger << "[SimulationFactory] Loaded image " << parameters.imageFile << " (" << image.getWidth() << "px x ";
    logger << image.getHeight() << "px)" << std::endl;

    std::random_device randomSeed;
    this->moveGenerator = std::unique_ptr<MoveGenerator>(new CauchyMoveGenerator(parameters.sigma, randomSeed()));
    this->imageBC = std::unique_ptr<ImageBoundaryConditions>(new WallBoundaryConditions());
    this->moveFilter = std::unique_ptr<MoveFilter>(new ImageMoveFilter(image, this->imageBC.get(), randomSeed()));
    Move drift = {parameters.driftX, parameters.driftY};
    this->randomWalker = std::unique_ptr<RandomWalker>(
        new RandomWalker(parameters.numberOfSteps, parameters.tracerRadius, drift, this->moveGenerator.get(),
                         this->moveFilter.get())
    );

    auto &imageMoveFilter = dynamic_cast<ImageMoveFilter&>(*this->moveFilter);
    logger << "[SimulationFactory] Found ";
    logger << imageMoveFilter.getNumberOfValidTracers(parameters.tracerRadius) << " valid starting points out of ";
    logger << imageMoveFilter.getNumberOfAllPoints() << std::endl;
}

RandomWalker& SimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
