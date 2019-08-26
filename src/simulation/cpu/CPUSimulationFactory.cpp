/*
 * CPUSimulationFactory.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <sstream>

#include "CPUSimulationFactory.h"
#include "move_generator/GaussianMoveGenerator.h"
#include "move_generator/CauchyMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/image_move_filter/ImageMoveFilter.h"
#include "move_filter/image_move_filter/WallBoundaryConditions.h"
#include "move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

std::unique_ptr<MoveGenerator> CPUSimulationFactory::createMoveGenerator(const Parameters& parameters) {
    std::istringstream moveGeneratorStream(parameters.moveGenerator);
    std::string moveGeneratorType;
    float sigma;
    moveGeneratorStream >> moveGeneratorType >> sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(sigma >= 0.f);

    if (moveGeneratorType == "GaussianMoveGenerator")
        return std::unique_ptr<MoveGenerator>(new GaussianMoveGenerator(sigma, this->seedGenerator()));
    else if (moveGeneratorType == "CauchyMoveGenerator")
        return std::unique_ptr<MoveGenerator>(new CauchyMoveGenerator(sigma, this->seedGenerator()));
    else
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorType);
}

std::unique_ptr<ImageBoundaryConditions>
CPUSimulationFactory::createImageBoundaryConditions(std::istringstream& moveFilterStream) {
    std::string imageBCType;
    moveFilterStream >> imageBCType;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    if (imageBCType == "WallBoundaryConditions")
        return std::unique_ptr<ImageBoundaryConditions>(new WallBoundaryConditions());
    else if (imageBCType == "PeriodicBoundaryConditions")
        return std::unique_ptr<ImageBoundaryConditions>(new PeriodicBoundaryConditions());
    else
        throw std::runtime_error("Unknown ImageBoundaryConditions: " + imageBCType);
}

std::unique_ptr<MoveFilter> CPUSimulationFactory::createImageMoveFilter(const Parameters& parameters,
                                                                        std::istringstream& moveFilterStream,
                                                                        std::ostream& logger) {
    std::string imageFilename;
    moveFilterStream >> imageFilename;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    std::ifstream imageFile(imageFilename);
    if (!imageFile)
        throw std::runtime_error("Cannot open " + imageFilename + " to load image");

    PPMImageReader imageReader;
    Image image = imageReader.read(imageFile);
    logger << "[CPUSimulationFactory] Loaded image " << imageFilename << " (" << image.getWidth() << "px x ";
    logger << image.getHeight() << "px)" << std::endl;

    this->imageBC = createImageBoundaryConditions(moveFilterStream);

    auto imageMoveFilter = new ImageMoveFilter(image, this->imageBC.get(), this->seedGenerator());
    logger << "[CPUSimulationFactory] Found " << imageMoveFilter->getNumberOfValidTracers(parameters.tracerRadius);
    logger << " valid starting points out of " << imageMoveFilter->getNumberOfAllPoints() << std::endl;
    return std::unique_ptr<MoveFilter>(imageMoveFilter);
}

std::unique_ptr<MoveFilter> CPUSimulationFactory::createMoveFilter(const Parameters& parameters, std::ostream& logger) {
    std::istringstream moveFilterStream(parameters.moveFilter);
    std::string moveFilterType;
    moveFilterStream >> moveFilterType;
    if (moveFilterType == "DefaultMoveFilter")
        return std::unique_ptr<MoveFilter>(new DefaultMoveFilter());
    else if (moveFilterType == "ImageMoveFilter")
        return createImageMoveFilter(parameters, moveFilterStream, logger);
    else
        throw std::runtime_error("Unknown MoveFilter: " + moveFilterType);
}

CPUSimulationFactory::CPUSimulationFactory(const Parameters &parameters, std::ostream &logger) {
    if (parameters.seed == "random") {
        unsigned long randomSeed = std::random_device()();
        this->seedGenerator.seed(randomSeed);
        logger << "[CPUSimulationFactory] Using random seed: " << randomSeed << std::endl;
    } else {
        this->seedGenerator.seed(std::stoul(parameters.seed));
    }

    this->moveGenerator = createMoveGenerator(parameters);
    this->moveFilter = createMoveFilter(parameters, logger);
    Move drift = {parameters.driftX, parameters.driftY};

    this->randomWalker.reset(new CPURandomWalker(parameters.numberOfWalks, parameters.numberOfSteps,
                                                 parameters.tracerRadius, drift, this->moveGenerator.get(),
                                                 this->moveFilter.get()));
}

RandomWalker &CPUSimulationFactory::getRandomWalker() {
    return *this->randomWalker;
}
