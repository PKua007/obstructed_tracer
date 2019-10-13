/*
 * PPMImageReader.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef PPMIMAGEREADER_H_
#define PPMIMAGEREADER_H_

#include <stdexcept>

#include "Image.h"

/**
 * @brief Exception thrown when parsing PPM file comes across some error.
 */
class ImageReadException : public std::runtime_error {
public:
    ImageReadException(const std::string &reason) : std::runtime_error{reason} { }
};

/**
 * @brief Class parsing binary pixelmap PPM 256-bit-per-channel image files.
 */
class PPMImageReader {
private:
    void verifyMetadata(std::istream& input, const std::string& magic, std::size_t width, std::size_t height,
                        std::size_t depth, int whitespaceBeforeData) const;
    Image readImageData(std::istream& input, std::size_t width, std::size_t height) const;

public:
    /**
     * @brief Reads PPM data from @a input stream and returns the Image.
     *
     * Only binary, 256-bit-per-color format is currently supported.
     *
     * @param input stream to read PPM data from
     * @return Image read from PPM data
     */
    Image read(std::istream &input) const;
};

#endif /* PPMIMAGEREADER_H_ */
