/*
 * PeriodicBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef PERIODICBOUNDARYCONDITIONS_H_
#define PERIODICBOUNDARYCONDITIONS_H_

#include "ImageMoveFilter.h"

class PeriodicBoundaryConditions : public ImageMoveFilter::ImageBoundaryConditions {
private:
    std::size_t width{};
    std::size_t height{};

    int mod(int a, int b) const { return (a % b + b) % b; }

public:
    void installOnImage(const Image &image) override;
    bool isImagePointInBounds(ImageMoveFilter::ImagePoint imagePoint, int radius) const override;
    ImageMoveFilter::ImagePoint applyOnImagePoint(ImageMoveFilter::ImagePoint imagePoint) const override;
};

#endif /* PERIODICBOUNDARYCONDITIONS_H_ */
