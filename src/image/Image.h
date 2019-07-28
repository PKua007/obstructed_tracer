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

using Color = unsigned int;

class Image {
private:
    std::size_t width;
    std::size_t height;
    std::vector<Color> data;

public:
    Image() = default;

    std::size_t getHeight() const;
    std::size_t getWidth() const;
    Color *getData();
    const Color *getData() const;
    Color &operator()(std::size_t x, std::size_t y);
    Color operator()(std::size_t x, std::size_t y) const;
};

#endif /* IMAGE_H_ */
