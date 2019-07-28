/*
 * PPMImageLoader.h
 *
 *  Created on: 29 lip 2019
 *      Author: pkua
 */

#ifndef PPMIMAGELOADER_H_
#define PPMIMAGELOADER_H_

#include "Image.h"

class PPMImageLoader {
public:
    Image load(std::istream &input) const;
};

#endif /* PPMIMAGELOADER_H_ */
