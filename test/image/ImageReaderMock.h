/*
 * ImageReaderMock.h
 *
 *  Created on: 24 gru 2019
 *      Author: pkua
 */

#ifndef IMAGEREADERMOCK_H_
#define IMAGEREADERMOCK_H_

#include <catch2/trompeloeil.hpp>

class ImageReaderMock : public ImageReader {
public:
     MAKE_CONST_MOCK1(read, Image(std::istream &), override);
};

#endif /* IMAGEREADERMOCK_H_ */
