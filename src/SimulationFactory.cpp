/*
 * SimulationFactory.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <sstream>

#include "SimulationFactory.h"
#include "move_generator/GaussianMoveGenerator.h"
#include "move_generator/CauchyMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/image_move_filter/ImageMoveFilter.h"
#include "move_filter/image_move_filter/WallBoundaryConditions.h"
#include "move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

SimulationFactory::SimulationFactory(Parameters parameters, std::ostream &logger) {

    std::random_device randomSeed;

    std::istringstream moveGeneratorStream(parameters.moveGenerator);
    std::string moveGeneratorType;
    float sigma;
    moveGeneratorStream >> moveGeneratorType >> sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(sigma >= 0.f);

    if (moveGeneratorType == "GaussianMoveGenerator")
        this->moveGenerator = std::unique_ptr<MoveGenerator>(new GaussianMoveGenerator(sigma, randomSeed()));
    else if (moveGeneratorType == "CauchyMoveGenerator")
        this->moveGenerator = std::unique_ptr<MoveGenerator>(new CauchyMoveGenerator(sigma, randomSeed()));
    else
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorType);

    std::istringstream moveFilterStream(parameters.moveFilter);
    std::string moveFilterType;
    moveFilterStream >> moveFilterType;

    if (moveFilterType == "DefaultMoveFilter") {
        this->moveFilter = std::unique_ptr<MoveFilter>(new DefaultMoveFilter());
    } else if (moveFilterType == "ImageMoveFilter") {
        std::string imageFilename;
        std::string imageBCType;
        moveFilterStream >> imageFilename >> imageBCType;
        if (!moveFilterStream)
            throw std::runtime_error("Malformed ImageMoveFilter parameters");

        std::ifstream imageFile(imageFilename);
        if (!imageFile)
            throw std::runtime_error("Cannot open " + imageFilename + " to load image");

        PPMImageReader imageReader;
        Image image = imageReader.read(imageFile);
        logger << "[SimulationFactory] Loaded image " << imageFilename << " (" << image.getWidth() << "px x ";
        logger << image.getHeight() << "px)" << std::endl;

        if (imageBCType == "WallBoundaryConditions")
            this->imageBC = std::unique_ptr<ImageBoundaryConditions>(new WallBoundaryConditions());
        else if (imageBCType == "PeriodicBoundaryConditions")
            this->imageBC = std::unique_ptr<PeriodicBoundaryConditions>(new PeriodicBoundaryConditions());
        else
            throw std::runtime_error("Unknown ImageBoundaryConditions: " + imageBCType);

        auto imageMoveFilter = new ImageMoveFilter(image, this->imageBC.get(), randomSeed());
        this->moveFilter = std::unique_ptr<MoveFilter>(imageMoveFilter);

        logger << "[SimulationFactory] Found ";
        logger << imageMoveFilter->getNumberOfValidTracers(parameters.tracerRadius) << " valid starting points out of ";
        logger << imageMoveFilter->getNumberOfAllPoints() << std::endl;
    } else {
        throw std::runtime_error("Unknown MoveFilter: " + moveFilterType);
    }

    Move drift = {parameters.driftX, parameters.driftY};
    this->randomWalker = std::unique_ptr<RandomWalker>(
        new RandomWalker(parameters.numberOfSteps, parameters.tracerRadius, drift, this->moveGenerator.get(),
                         this->moveFilter.get())
    );
}

RandomWalker& SimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
