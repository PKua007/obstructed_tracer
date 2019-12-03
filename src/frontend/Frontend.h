/*
 * Frontend.h
 *
 *  Created on: 2 gru 2019
 *      Author: pkua
 */

#ifndef FRONTEND_H_
#define FRONTEND_H_

#include <vector>
#include <iosfwd>

#include "Parameters.h"

class Frontend {
private:
    std::string command;
    std::string mode;
    Parameters parameters;
    std::vector<std::string> additionalArguments;

    int perform_walk();
    int analyze();

public:
    virtual ~Frontend() { };

    Frontend(int argc, char **argv);

    int run();
};

#endif /* FRONTEND_H_ */
