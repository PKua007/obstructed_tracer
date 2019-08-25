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
#include "image/Image.h"
#include "ImageBoundaryConditions.h"
#include "ImagePoint.h"

class ImageMoveFilter: public MoveFilter {
private:
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    std::size_t width;
    std::size_t height;
    ImageBoundaryConditions *imageBC;
    std::vector<bool> validPointsMap;

    float radiusForTracerCache = -1.0;
    std::vector<std::size_t> validTracerIndicesCache;

    void rebuildValidTracersCache(float radius);
    bool isPointValid(ImagePoint point, float pointRadius) const;
    bool checkValidPointsMap(ImagePoint point) const;
    bool isLineValid(ImagePoint from, ImagePoint to, float pointRadius) const;
    ImagePoint indexToPoint(std::size_t index) const;

public:
    ImageMoveFilter(Image image, ImageBoundaryConditions *imageBC, unsigned int seed);

    bool isMoveValid(Tracer tracer, Move move) const override;
    Tracer randomValidTracer(float radius) override;

    std::size_t getNumberOfAllPoints() const;
    std::size_t getNumberOfValidTracers(float radius);
};

#endif /* IMAGEMOVEFILTER_H_ */
