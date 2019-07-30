/*
 * PPMImageReader.cpp
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#include <istream>

#include "PPMImageReader.h"

namespace {
    class CommentStripper {

    };

    std::istream &operator>>(std::istream &stream, CommentStripper stripper) {
        std::istream::sentry sEntry(stream);
        if (!sEntry)
            return stream;

        stream >> std::ws;
        while (stream.good() && stream.peek() == '#')
            while (stream.good() && stream.get() != '\n')
                { }

        return stream;
    }
}

void PPMImageReader::verifyMetadata(std::istream& input, const std::string& magic, std::size_t width, std::size_t height,
                                std::size_t depth, int whitespaceBeforeData) const {
    if (!input)
        throw ImageReadException("[PPMImageReader::read] Malformed PPM file");
    if (magic != "P6")
        throw ImageReadException("[PPMImageReader::read] Only P6 (binary PPM) format supported");
    if (width == 0)
        throw ImageReadException("[PPMImageReader::read] width == 0");
    if (height == 0)
        throw ImageReadException("[PPMImageReader::read] height == 0");
    if (depth != 255)
        throw ImageReadException("[PPMImageReader::read] Only 255 maximal depth supported");
    if (!std::isspace(whitespaceBeforeData))
        throw ImageReadException("[PPMImageReader::read] Malformed PPM file");
}

Image PPMImageReader::readImageData(std::istream& input, std::size_t width, std::size_t height) const {
    Image image(width, height);
    for (std::size_t j = 0; j < height; j++) {
        for (std::size_t i = 0; i < width; i++) {
            Color pixel;
            input.read((char*) ((&pixel)), 3);
            if (!input)
                throw ImageReadException("[PPMImageReader::read] Unexpected rastor data end");
            image(i, j) = pixel;
        }
    }
    return image;
}

Image PPMImageReader::read(std::istream& input) const {
    CommentStripper stripper;
    std::string magic;
    std::size_t width, height, depth;
    int whitespaceBeforeData;

    input >> stripper >> magic >> stripper >> width >> stripper >> height >> stripper >> depth;
    whitespaceBeforeData = input.get();
    verifyMetadata(input, magic, width, height, depth, whitespaceBeforeData);

    return readImageData(input, width, height);
}
