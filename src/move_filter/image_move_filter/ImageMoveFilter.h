/*
 * ImageMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGEMOVEFILTER_H_
#define IMAGEMOVEFILTER_H_

#include <random>

#include "simulation/MoveFilter.h"
#include "ImageBoundaryConditions.h"
#include "ImagePoint.h"

class ImageMoveFilter: public MoveFilter {
private:
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    size_t width{};
    size_t height{};
    ImageBoundaryConditions *imageBC{};
    bool* validPointsMap{};
    size_t validPointsMapSize{};

    float radiusForTracerCache = -1.0;
    size_t *validTracerIndicesCache{};
    size_t validTracerIndicesCacheSize{};

    ImageMoveFilter(const ImageMoveFilter &other) { }

    void rebuildValidTracersCache(float radius);
    bool isPointValid(ImagePoint point, float pointRadius) const;
    bool checkValidPointsMap(ImagePoint point) const;
    bool isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const;
    ImagePoint indexToPoint(std::size_t index) const;
    size_t pointToIndex(ImagePoint point) const;

public:
    ImageMoveFilter(unsigned int *intImageData, size_t width, size_t height, ImageBoundaryConditions *imageBC,
                    unsigned long seed);
    ~ImageMoveFilter();

    bool isMoveValid(Tracer tracer, Move move) const override;
    Tracer randomValidTracer(float radius) override;

    size_t getNumberOfAllPoints() const;
    size_t getNumberOfValidTracers(float radius);
};

#endif /* IMAGEMOVEFILTER_H_ */
