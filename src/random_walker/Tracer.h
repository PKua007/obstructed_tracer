/*
 * Tracer.h
 *
 *  Created on: 30 lip 2019
 *      Author: pkua
 */

#ifndef TRACER_H_
#define TRACER_H_

#include "../Point.h"

class Tracer {
private:
    Point position{};
    float radius{};

public:
    Tracer() = default;
    Tracer(Point position, float radius) : position{position}, radius{radius} { }

    Point getPosition() const { return position; }
    void setPosition(Point position) { this->position = position; }
    float getRadius() const { return radius; }
    void setRadius(float radius) { this->radius = radius; }

    Tracer &operator+=(Move move) {
        this->position += move;
        return *this;
    }
};

#endif /* TRACER_H_ */
