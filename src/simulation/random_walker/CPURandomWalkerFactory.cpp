/*
 * CPURandomWalkerFactory.cpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <sstream>

#include "CPURandomWalkerFactory.h"
#include "simulation/move_generator/cpu/CPUGaussianMoveGenerator.h"
#include "simulation/move_generator/cpu/CPUCauchyMoveGenerator.h"
#include "simulation/move_filter/DefaultMoveFilter.h"
#include "simulation/move_filter/image_move_filter/ImageMoveFilter.h"
#include "simulation/move_filter/image_move_filter/WallBoundaryConditions.h"
#include "simulation/move_filter/image_move_filter/PeriodicBoundaryConditions.h"
#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

std::unique_ptr<MoveGenerator> CPURandomWalkerFactory::createMoveGenerator(const std::string &moveGeneratorParameters) {
    std::istringstream moveGeneratorStream(moveGeneratorParameters);
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

std::unique_ptr<MoveFilter> CPURandomWalkerFactory::createImageMoveFilter(std::istringstream &moveFilterStream,
                                                                          std::ostream &logger)
{
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

    std::string imageBCType;
    moveFilterStream >> imageBCType;
    if (!moveFilterStream)
        throw std::runtime_error("Malformed ImageMoveFilter parameters");

    if (imageBCType == "WallBoundaryConditions") {
        return std::unique_ptr<ImageMoveFilter<WallBoundaryConditions>>(
            new ImageMoveFilter<WallBoundaryConditions>(imageData.data(), image.getWidth(), image.getHeight(),
                                                        this->seedGenerator(), this->numberOfWalksInSeries)
        );
    } else if (imageBCType == "PeriodicBoundaryConditions") {
        return std::unique_ptr<ImageMoveFilter<PeriodicBoundaryConditions>>(
            new ImageMoveFilter<PeriodicBoundaryConditions>(imageData.data(), image.getWidth(), image.getHeight(),
                                                            this->seedGenerator(), this->numberOfWalksInSeries)
        );
    } else {
        throw std::runtime_error("Unknown ImageBoundaryConditions: " + imageBCType);
    }
}

std::unique_ptr<MoveFilter> CPURandomWalkerFactory::createMoveFilter(const std::string &moveFilterParameters,
                                                                     std::ostream &logger)
{
    std::istringstream moveFilterStream(moveFilterParameters);
    std::string moveFilterType;
    moveFilterStream >> moveFilterType;
    if (moveFilterType == "DefaultMoveFilter")
        return std::unique_ptr<MoveFilter>(new DefaultMoveFilter());
    else if (moveFilterType == "ImageMoveFilter")
        return createImageMoveFilter(moveFilterStream, logger);
    else
        throw std::runtime_error("Unknown MoveFilter: " + moveFilterType);
}

CPURandomWalkerFactory::CPURandomWalkerFactory(unsigned long seed, const WalkerParameters &walkerParameters,
                                               std::ostream &logger)
        : walkerParameters{walkerParameters}, numberOfWalksInSeries{walkerParameters.numberOfWalksInSeries},
          logger{logger}, seedGenerator(seed)
{ }

std::unique_ptr<RandomWalker> CPURandomWalkerFactory::createRandomWalker() {
    auto moveGenerator = this->createMoveGenerator(walkerParameters.moveGeneratorParameters);
    auto moveFilter = this->createMoveFilter(walkerParameters.moveFilterParameters, logger);

    return std::unique_ptr<RandomWalker>(
        new CPURandomWalker(this->numberOfWalksInSeries, this->walkerParameters.walkParameters,
                            std::move(moveGenerator), std::move(moveFilter), this->logger)
    );
}
