/*
 * Parameters.cpp
 *
 *  Created on: 28 lip 2019
 *      Author: pkua
 */

#include <iostream>

#include "Parameters.h"
#include "Config.h"
#include "utils/Assertions.h"

Parameters::Parameters(std::istream& input) {
    auto config = Config::parse(input, '=', true);

    for (const auto &key : config.getKeys()) {
        if (key == "numberOfSteps")
            this->numberOfSteps = config.getUnsignedLong(key);
        else if (key == "tracerRadius")
            this->tracerRadius = config.getFloat(key);
        else if (key == "moveFilter")
            this->moveFilter = config.getString(key);
        else if (key == "moveGenerator")
            this->moveGenerator = config.getString(key);
        else if (key == "driftX")
            this->driftX = config.getFloat(key);
        else if (key == "driftY")
            this->driftY = config.getFloat(key);
        else if (key == "numberOfWalksInSeries")
            this->numberOfWalksInSeries = config.getUnsignedLong(key);
        else if (key == "numberOfSeries")
            this->numberOfSeries = config.getUnsignedLong(key);
        else if (key == "storeTrajectories")
            this->storeTrajectories = (config.getString(key) == "true" ? true : false);
        else if (key == "seed")
            this->seed = config.getString(key);
        else if (key == "device")
            this->device = config.getString(key);
        else
            std::cerr << "[Parameters::Parameters] Warning: unknown parameter " << key << std::endl;
    }

    this->validateParameters();
}

void Parameters::print(std::ostream& out) {
    out << "numberOfSteps         : " << this->numberOfSteps << std::endl;
    out << "tracerRadius          : " << this->tracerRadius << std::endl;
    out << "moveGenerator         : " << this->moveGenerator << std::endl;
    out << "moveFilter            : " << this->moveFilter << std::endl;
    out << "driftX                : " << this->driftX << std::endl;
    out << "driftY                : " << this->driftY << std::endl;
    out << "numberOfWalksInSeries : " << this->numberOfWalksInSeries << std::endl;
    out << "numberOfSeries        : " << this->numberOfSeries << std::endl;
    out << "storeTrajectories     : " << (this->storeTrajectories ? "true" : "false") << std::endl;
    out << "seed                  : " << this->seed << std::endl;
    out << "device                : " << this->device << std::endl;
}

void Parameters::validateParameters() const {
    Validate(this->numberOfSteps > 0);
    Validate(this->tracerRadius >= 0.f);
    Validate(this->numberOfWalksInSeries > 0);
    Validate(this->numberOfSeries > 0);
}
