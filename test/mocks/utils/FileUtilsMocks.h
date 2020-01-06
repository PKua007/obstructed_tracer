/*
 * FileUtilsMocks.h
 *
 *  Created on: 24 gru 2019
 *      Author: pkua
 */

#ifndef FILEUTILSMOCKS_H_
#define FILEUTILSMOCKS_H_

#include <catch2/trompeloeil.hpp>

#include "utils/FileUtils.h"

class FileOstreamProviderMock : public FileOstreamProvider {
public:
    MAKE_CONST_MOCK1(openFile, std::unique_ptr<std::ostream>(const std::string &), override);
};

class FileIstreamProviderMock : public FileIstreamProvider {
public:
    MAKE_CONST_MOCK1(openFile, std::unique_ptr<std::istream>(const std::string &), override);
};

#endif /* FILEUTILSMOCKS_H_ */
