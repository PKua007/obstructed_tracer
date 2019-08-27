/*
 * WallBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef WALLBOUNDARYCONDITIONS_H_
#define WALLBOUNDARYCONDITIONS_H_

#include "ImageBoundaryConditions.h"

class WallBoundaryConditions: public ImageBoundaryConditions {
private:
    std::size_t width{};
    std::size_t height{};

public:
    void setupDimensions(size_t width, size_t height) override;
    bool isImagePointInBounds(ImagePoint imagePoint, int radius) const override;
    ImagePoint applyOnImagePoint(ImagePoint imagePoint) const override;
};

#endif /* WALLBOUNDARYCONDITIONS_H_ */
