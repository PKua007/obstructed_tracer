/*
 * Point.h
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#ifndef POINT_H_
#define POINT_H_

#include <iosfwd>

struct Point  {
    float x;
    float y;
};

std::ostream &operator<<(std::ostream &out, Point point);

#endif /* POINT_H_ */
