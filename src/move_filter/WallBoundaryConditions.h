/*
 * WallBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef WALLBOUNDARYCONDITIONS_H_
#define WALLBOUNDARYCONDITIONS_H_

#include "ImageMoveFilter.h"

class WallBoundaryConditions: public ImageMoveFilter::ImageBoundaryConditions {
private:
    std::size_t width{};
    std::size_t height{};

public:
    void installOnImage(const Image &image) override;
    bool isImagePointInBounds(ImageMoveFilter::ImagePoint imagePoint, int radius) const override;
    ImageMoveFilter::ImagePoint applyOnImagePoint(ImageMoveFilter::ImagePoint imagePoint) const override;
};

#endif /* WALLBOUNDARYCONDITIONS_H_ */
