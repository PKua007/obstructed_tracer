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
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution{0.f, 1.f};
    Image image;

public:
    ImageMoveFilter(Image image, unsigned int seed);

    bool isMoveValid(Point tracer, Move move) const;
    Point randomValidPoint() const;
};

#endif /* IMAGEMOVEFILTER_H_ */
