/*
 * CPURandomWalkerFactory.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <sstream>

#include "CPURandomWalkerFactory.h"
#include "move_generator/cpu/CPUGaussianMoveGenerator.h"
#include "move_generator/cpu/CPUCauchyMoveGenerator.h"
#include "move_filter/DefaultMoveFilter.h"
#include "move_filter/image_move_filter/ImageMoveFilter.h"
#include "move_filter/image_move_filter/WallBoundaryConditions.h"
#include "move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

std::unique_ptr<MoveGenerator> CPURandomWalkerFactory::createMoveGenerator(const Parameters& parameters) {
    std::istringstream moveGeneratorStream(parameters.moveGenerator);
    std::string moveGeneratorType;
    float sigma;
    moveGeneratorStream >> moveGeneratorType >> sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(sigma >= 0.f);

    if (moveGeneratorType == "GaussianMoveGenerator")
        return std::unique_ptr<MoveGenerator>(new CPUGaussianMoveGenerator(sigma, this->seedGenerator()));
    else if (moveGeneratorType == "CauchyMoveGenerator")
        return std::unique_ptr<MoveGenerator>(new CPUCauchyMoveGenerator(sigma, this->seedGenerator()));
    else
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorType);
}

std::unique_ptr<ImageBoundaryConditions>
CPURandomWalkerFactory::createImageBoundaryConditions(std::istringstream& moveFilterStream) {
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

std::unique_ptr<MoveFilter> CPURandomWalkerFactory::createImageMoveFilter(const Parameters& parameters,
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
    auto imageData = image.getIntData();
    logger << "[CPURandomWalkerFactory] Loaded image " << imageFilename << " (" << image.getWidth() << "px x ";
    logger << image.getHeight() << "px)" << std::endl;

    this->imageBC = createImageBoundaryConditions(moveFilterStream);

    auto imageMoveFilter = new ImageMoveFilter(imageData.data(), image.getWidth(), image.getHeight(),
                                               this->imageBC.get(), this->seedGenerator(),
                                               parameters.numberOfWalksInSeries);
    logger << "[CPURandomWalkerFactory] Found " << imageMoveFilter->getNumberOfValidTracers();
    logger << " valid starting points out of " << imageMoveFilter->getNumberOfAllPoints() << std::endl;
    return std::unique_ptr<MoveFilter>(imageMoveFilter);
}

std::unique_ptr<MoveFilter> CPURandomWalkerFactory::createMoveFilter(const Parameters& parameters,
                                                                     std::ostream& logger) {
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

CPURandomWalkerFactory::CPURandomWalkerFactory(unsigned long seed, const Parameters &parameters, std::ostream &logger) {
    this->seedGenerator.seed(seed);
    this->moveGenerator = this->createMoveGenerator(parameters);
    this->moveFilter = this->createMoveFilter(parameters, logger);

    Move drift = {parameters.driftX, parameters.driftY};
    RandomWalker::WalkParameters walkParameters = {parameters.numberOfSteps, parameters.tracerRadius, drift};
    this->randomWalker.reset(new CPURandomWalker(parameters.numberOfWalksInSeries, walkParameters,
                                                 this->moveGenerator.get(), this->moveFilter.get(), logger));
}

RandomWalker &CPURandomWalkerFactory::getRandomWalker() {
    return *this->randomWalker;
}
