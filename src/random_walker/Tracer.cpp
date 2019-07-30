/*
 * Tracer.cpp
 *
 *  Created on: 30 lip 2019
 *      Author: pkua
 */

#include "Tracer.h"

Tracer &Tracer::operator+=(Move move) {
    this->position += move;
    return *this;
}
