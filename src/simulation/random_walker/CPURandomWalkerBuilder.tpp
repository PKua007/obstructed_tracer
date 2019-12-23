/*
 * CPURandomWalkerFactory.tpp
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#include <fstream>
#include <sstream>

#include "image/PPMImageReader.h"
#include "utils/Assertions.h"

template<typename CPURandomWalker_t>
std::unique_ptr<MoveGenerator>
CPURandomWalkerBuilder<CPURandomWalker_t>::createMoveGenerator(const std::string &moveGeneratorParameters,
                                                               float integrationStep)
{
    std::istringstream moveGeneratorStream(moveGeneratorParameters);
    std::string moveGeneratorType;
    float sigma;
    moveGeneratorStream >> moveGeneratorType >> sigma;
    if (!moveGeneratorStream)
        throw std::runtime_error("Malformed MoveGenerator parameters");
    Validate(sigma >= 0.f);

    if (moveGeneratorType == "GaussianMoveGenerator") {
        return std::unique_ptr<MoveGenerator>(
            new GaussianMoveGenerator_t(sigma, integrationStep, this->seedGenerator())
        );
    } else if (moveGeneratorType == "CauchyMoveGenerator") {
        return std::unique_ptr<MoveGenerator>(
            new CauchyMoveGenerator_t(sigma, integrationStep, this->seedGenerator())
        );
    } else{
        throw std::runtime_error("Unknown MoveGenerator: " + moveGeneratorType);
    }
}

template<typename CPURandomWalker_t>
std::unique_ptr<MoveFilter>
CPURandomWalkerBuilder<CPURandomWalker_t>::createImageMoveFilter(std::istringstream &moveFilterStream,
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
        return std::unique_ptr<MoveFilter>(
            new ImageMoveFilterWallBC_t(imageData.data(), image.getWidth(), image.getHeight(), this->seedGenerator(),
                                        this->numberOfWalksInSeries)
        );
    } else if (imageBCType == "PeriodicBoundaryConditions") {
        return std::unique_ptr<MoveFilter>(
            new ImageMoveFilterPeriodicBC_t(imageData.data(), image.getWidth(), image.getHeight(),
                                            this->seedGenerator(), this->numberOfWalksInSeries)
        );
    } else {
        throw std::runtime_error("Unknown ImageBoundaryConditions: " + imageBCType);
    }
}

template<typename CPURandomWalker_t>
std::unique_ptr<MoveFilter>
CPURandomWalkerBuilder<CPURandomWalker_t>::createMoveFilter(const std::string &moveFilterParameters,
                                                            std::ostream &logger)
{
    std::istringstream moveFilterStream(moveFilterParameters);
    std::string moveFilterType;
    moveFilterStream >> moveFilterType;
    if (moveFilterType == "DefaultMoveFilter")
        return std::unique_ptr<MoveFilter>(new DefaultMoveFilter_t());
    else if (moveFilterType == "ImageMoveFilter")
        return createImageMoveFilter(moveFilterStream, logger);
    else
        throw std::runtime_error("Unknown MoveFilter: " + moveFilterType);
}

template<typename CPURandomWalker_t>
CPURandomWalkerBuilder<CPURandomWalker_t>::CPURandomWalkerBuilder(unsigned long seed,
                                                                  const RandomWalkerFactory::WalkerParameters &
                                                                  walkerParameters, std::ostream &logger)
        : walkerParameters{walkerParameters}, numberOfWalksInSeries{walkerParameters.numberOfWalksInSeries},
          logger{logger}, seedGenerator(seed)
{ }

template<typename CPURandomWalker_t>
std::unique_ptr<RandomWalker> CPURandomWalkerBuilder<CPURandomWalker_t>::build() {
    float integrationStep = this->walkerParameters.walkParameters.integrationStep;
    auto moveGenerator = this->createMoveGenerator(walkerParameters.moveGeneratorParameters, integrationStep);
    auto moveFilter = this->createMoveFilter(walkerParameters.moveFilterParameters, logger);

    return std::unique_ptr<RandomWalker>(
        new CPURandomWalker_t(this->numberOfWalksInSeries, this->walkerParameters.walkParameters,
                              std::move(moveGenerator), std::move(moveFilter), this->logger)
    );
}
