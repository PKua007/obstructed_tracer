/*
 * Image.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <sstream>
#include <iomanip>

#include "Image.h"
#include "utils/Assertions.h"


Image::Image(std::size_t width, std::size_t height) :
        width{width}, height{height} {
    this->data.resize(width * height, WHITE);
}

std::size_t Image::getHeight() const {
    return this->height;
}

std::size_t Image::getWidth() const {
    return this->width;
}

std::size_t Image::getNumberOfPixels() const {
    return this->width * this->height;
}


std::vector<uint32_t> Image::getIntData() const {
    return std::vector<uint32_t>(this->data.begin(), this->data.end());
}

Color& Image::operator()(std::size_t x, std::size_t y) {
    Expects(x < this->width);
    Expects(y < this->height);
    return this->data[x + y * this->width];
}

Color Image::operator()(std::size_t x, std::size_t y) const {
    Expects(x < this->width);
    Expects(y < this->height);
    return this->data[x + y * this->width];
}

std::ostream& operator<<(std::ostream& out, Color color) {
    uint32_t hexColor = static_cast<uint32_t>(color);
    std::ostringstream sStream;
    sStream << "0x" << std::setfill('0') << std::setw(6) << std::hex << hexColor;
    out << sStream.str();
    return out;
}

bool operator==(Color color1, Color color2) {
    return color1.r == color2.r && color1.g == color2.g && color1.b == color2.b;
}

bool operator!=(Color color1, Color color2) {
    return !(color1 == color2);
}
