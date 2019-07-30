/*
 * Point.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef POINT_H_
#define POINT_H_

#include <iosfwd>

#include "Move.h"

struct Point  {
    float x{};
    float y{};

    Point() = default;
    Point(float x, float y) : x{x}, y{y} { }

    Point operator+(Move move) const;
    Point operator-(Move move) const;
    Point &operator+=(Move move);
    Point &operator-=(Move move);
};

std::ostream &operator<<(std::ostream &out, Point point);

#endif /* POINT_H_ */
