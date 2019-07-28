/*
 * Image.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include "Image.h"


std::size_t Image::getHeight() const {
    return this->height;
}

std::size_t Image::getWidth() const {
    return this->width;
}


Color* Image::getData() {
    return this->data.data();
}

const Color* Image::getData() const {
    return this->data.data();
}

Color& Image::operator ()(std::size_t x, std::size_t y) {
    return this->data[x + y * this->width];
}

Color Image::operator ()(std::size_t x, std::size_t y) const {
    return this->data[x + y * this->width];
}
