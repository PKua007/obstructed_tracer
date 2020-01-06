/*
 * Image.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

/** @file */

#ifndef IMAGE_H_
#define IMAGE_H_

#include <vector>
#include <iosfwd>
#include <cstdint>

/**
 * @brief A simple struct representing a color.
 * @see operator<<(std::ostream &, Color)
 * @see operator==(Color, Color)
 * @see operator!=(Color, Color)
 */
struct Color {
    uint8_t r{};
    uint8_t g{};
    uint8_t b{};

    /**
     * @brief Default constructor - makes black.
     */
    Color() { }

    /**
     * @brief Constructs a color from individual channel values.
     */
    Color(uint8_t r, uint8_t g, uint8_t b) : r{r}, g{g}, b{b} { }

    /**
     * @brief Constructs a color from int data.
     * @param intColor a colot in 0xFFFFFFFF-like form, where alpha channel is ignored
     */
    Color(uint32_t intColor)
            : r{static_cast<uint8_t>(intColor >> 24)},
              g{static_cast<uint8_t>(intColor >> 16)},
              b{static_cast<uint8_t>(intColor >> 8)}
    { }

    /**
     * @brief Converts a color to 0xFFFFFFFF-like form with alpha channel always being 0xFF.
     * @return a color to 0xFFFFFFFF-like form with alpha channel always being 0xFF
     */
    operator uint32_t() const { return (this->r << 24) | (this->g << 16) | (this->b << 8) | 0xff; }
};

#define WHITE Color{255, 255, 255}
#define BLACK Color{0, 0, 0}

/**
 * @brief Stream insertion operator for Color printing in 0xFFFFFF style.
 * @param out stream to print @a color to
 * @param color Color to be printed
 * @return reference to @a out
 */
std::ostream &operator<<(std::ostream &out, Color color);

/**
 * @brief Checks if @a color1 and @a color2 are equal.
 */
bool operator==(Color color1, Color color2);

/**
 * @brief Checks if @a color1 and @a color2 are not equal.
 */
bool operator!=(Color color1, Color color2);

/**
 * @brief A class representing an RGB 24-bit image.
 */
class Image {
private:
    std::size_t width{};
    std::size_t height{};
    std::vector<Color> data;

public:
    /**
     * @brief Creates useless 0x0 image.
     */
    Image() : Image{0, 0} { }

    /**
     * @brief Created image of size @a witdh x @a height
     * @param width width of the image
     * @param height height of the image
     */
    Image(std::size_t width, std::size_t height);

    /**
     * @brief Returns the height of the image.
     * @return the height of the image
     */
    std::size_t getHeight() const;

    /**
     * @brief Returns the width of the image.
     * @return the width of the image
     */
    std::size_t getWidth() const;

    /**
     * @brief Returns total number of pixels in the image, namely Image::getWidth() * Image::getHeight()
     * @return total number of pixels in the image
     */
    std::size_t getNumberOfPixels() const;

    /**
     * @brief Returns a flattened array of 0xFFFFFFFF-like pixels row by row starting from the top.
     */
    std::vector<uint32_t> getIntData() const;

    /**
     * @brief Mutable access to pixels.
     *
     * `x == 0` and `y == 0` is top-left pixel and their go respectively right and down.
     *
     * @param x x coordinate of the pixel
     * @param y y coordinate of the pixel
     * @return pixel at `(x, y)` as Color struct
     */
    Color &operator()(std::size_t x, std::size_t y);

    /**
     * @brief Immutable access to pixels.
     *
     * `x == 0` and `y == 0` is top-left pixel and their go respectively right and down.
     *
     * @param x x coordinate of the pixel
     * @param y y coordinate of the pixel
     * @return pixel at `(x, y)` as Color struct
     */
    Color operator()(std::size_t x, std::size_t y) const;
};

#endif /* IMAGE_H_ */
