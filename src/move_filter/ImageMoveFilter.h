/*
 * ImageMoveFilter.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGEMOVEFILTER_H_
#define IMAGEMOVEFILTER_H_

#include <random>

#include "../random_walker/MoveFilter.h"
#include "../image/Image.h"

class ImageMoveFilter: public MoveFilter {
private:
    struct ImagePoint {
        std::size_t x;
        std::size_t y;
    };

    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};

    std::size_t width;
    std::size_t height;
    std::vector<bool> validPointsMap;
    std::vector<std::size_t> validPointsIndices;

    bool isPointValid(ImagePoint point) const;
    ImagePoint indexToPoint(std::size_t index) const;

public:
    ImageMoveFilter(Image image, unsigned int seed);

    bool isMoveValid(Point tracer, Move move) const;
    Point randomValidPoint();
};

#endif /* IMAGEMOVEFILTER_H_ */
