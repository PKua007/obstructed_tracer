/*
 * Image.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <vector>
#include <iosfwd>
#include <cstdint>

struct Color {
    uint8_t r;
    uint8_t g;
    uint8_t b;
};

#define WHITE Color{uint8_t{255}, uint8_t{255}, uint8_t{255}}
#define BLACK Color{uint8_t{255}, uint8_t{255}, uint8_t{255}}

std::ostream &operator<<(std::ostream &out, Color color);
bool operator==(Color color1, Color color2);

class Image {
private:
    std::size_t width{};
    std::size_t height{};
    std::vector<Color> data;

public:
    Image(std::size_t width, std::size_t height);

    std::size_t getHeight() const;
    std::size_t getWidth() const;
    Color *getData();
    const Color *getData() const;
    Color &operator()(std::size_t x, std::size_t y);
    Color operator()(std::size_t x, std::size_t y) const;
};

#endif /* IMAGE_H_ */
