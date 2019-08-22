/*
 * ImagePoint.h
 *
 *  Created on: 22 sie 2019
 *      Author: pkua
 */

#ifndef IMAGEPOINT_H_
#define IMAGEPOINT_H_

#include "../random_walker/Point.h"

struct ImagePoint {
    int x{};
    int y{};

    ImagePoint() = default;
    ImagePoint(int x, int y) : x{x}, y{y} { };
    ImagePoint(Point point) : x{static_cast<int>(point.x)}, y{static_cast<int>(point.y)} { };

    bool operator==(ImagePoint second) const { return this->x == second.x && this->y == second.y; }
    bool operator!=(ImagePoint second) const { return !(*this == second); }
};

#endif /* IMAGEPOINT_H_ */
