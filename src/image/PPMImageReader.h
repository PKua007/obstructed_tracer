/*
 * PPMImageReader.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef PPMIMAGEREADER_H_
#define PPMIMAGEREADER_H_

#include <stdexcept>

#include "ImageReader.h"

/**
 * @brief Class parsing binary pixelmap PPM 256-bit-per-channel image files.
 */
class PPMImageReader : public ImageReader {
private:
    void verifyMetadata(std::istream &input, const std::string &magic, std::size_t width, std::size_t height,
                        std::size_t depth, int whitespaceBeforeData) const;
    Image readImageData(std::istream &input, std::size_t width, std::size_t height) const;

public:
    /**
     * @brief Reads PPM data from @a input stream and returns the Image.
     *
     * Only binary, 256-bit-per-color format is currently supported.
     *
     * @param input stream to read PPM data from
     * @return Image read from PPM data
     */
    Image read(std::istream &input) const override;
};

#endif /* PPMIMAGEREADER_H_ */
