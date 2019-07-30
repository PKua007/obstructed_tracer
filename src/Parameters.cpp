/*
 * Parameters.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <iostream>

#include "Parameters.h"
#include "Config.h"

Parameters::Parameters(std::istream& input) {
    auto config = Config::parse(input, '=', true);

    for (const auto &key : config.getKeys()) {
        if (key == "sigma")
            this->sigma = config.getFloat(key);
        else if (key == "numberOfSteps")
            this->numberOfSteps = config.getUnsignedLong(key);
        else if (key == "imageFile")
            this->imageFile = config.getString(key);
        else
            std::cerr << "[Parameters::Parameters] Warning: unknown parameter " << key << std::endl;
    }
}

void Parameters::print(std::ostream& out) {
    out << "sigma         : " << this->sigma << std::endl;
    out << "numberOfSteps : " << this->numberOfSteps << std::endl;
    out << "imageFile     : " << this->imageFile << std::endl;
}
