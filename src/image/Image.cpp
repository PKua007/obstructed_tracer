/*
 * Image.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <sstream>
#include <iomanip>

#include "Image.h"
#include "../utils/Assertions.h"


Image::Image(std::size_t width, std::size_t height) :
        width{width}, height{height} {
    Expects(width > 0);
    Expects(height > 0);
    this->data.reserve(width * height);
}

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
    Expects(x < width);
    Expects(y < height);
    return this->data[x + y * this->width];
}

Color Image::operator ()(std::size_t x, std::size_t y) const {
    Expects(x < width);
    Expects(y < height);
    return this->data[x + y * this->width];
}

std::ostream& operator<<(std::ostream& out, Color color) {
    uint32_t hexColor = (color.r << 16) | (color.g << 8) | color.b;
    std::ostringstream sStream;
    sStream << "0x" << std::setfill('0') << std::setw(6) << std::hex << hexColor;
    out << sStream.str();
    return out;
}

bool operator ==(Color color1, Color color2) {
    return color1.r == color2.r && color1.g == color2.g && color1.b == color2.b;
}
