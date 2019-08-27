/*
 * PeriodicBoundaryConditions.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef PERIODICBOUNDARYCONDITIONS_H_
#define PERIODICBOUNDARYCONDITIONS_H_

#include "ImageBoundaryConditions.h"

class PeriodicBoundaryConditions : public ImageBoundaryConditions {
private:
    size_t width{};
    size_t height{};

    int mod(int a, int b) const { return (a % b + b) % b; }

public:
    void setupDimensions(size_t width, size_t height) override;
    bool isImagePointInBounds(ImagePoint imagePoint, int radius) const override;
    ImagePoint applyOnImagePoint(ImagePoint imagePoint) const override;
};

#endif /* PERIODICBOUNDARYCONDITIONS_H_ */
