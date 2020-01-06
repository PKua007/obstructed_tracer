/*
 * ImageReader.h
 *
 *  Created on: 24 gru 2019
 *      Author: pkua
 */

#ifndef IMAGEREADER_H_
#define IMAGEREADER_H_


#include "Image.h"

/**
 * @brief Exception thrown when parsing PPM file comes across some error.
 */
class ImageReadException : public std::runtime_error {
public:
    ImageReadException(const std::string &reason) : std::runtime_error{reason} { }
};


/**
 * @brief Class parsing image files.
 */
class ImageReader {
public:
    virtual ~ImageReader() { }

    /**
     * @brief Reads image data from @a input stream and returns the Image.
     * @param input stream to read image data from
     * @return Image read from image data
     */
    virtual Image read(std::istream &input) const = 0;
};

#endif /* IMAGEREADER_H_ */
