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

class ImageReadException : public std::runtime_error {
public:
    ImageReadException(const std::string &reason) : std::runtime_error{reason} { }
};

class PPMImageReader {
public:
    Image read(std::istream &input) const;
};

#endif /* PPMIMAGEREADER_H_ */
