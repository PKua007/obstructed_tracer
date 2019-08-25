/*
 * Point.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <ostream>

#include "Point.h"

std::ostream &operator<<(std::ostream& out, Point point) {
    out << "{" << point.x << ", " << point.y << "}";
    return out;
}

Point Point::operator+(Move move) const {
    return {this->x + move.x, this->y + move.y};
}

Point Point::operator-(Move move) const {
    return {this->x - move.x, this->y - move.y};
}

Point& Point::operator+=(Move move) {
    this->x += move.x;
    this->y += move.y;
    return *this;
}

Point& Point::operator-=(Move move) {
    this->x -= move.x;
    this->y -= move.y;
    return *this;
}
