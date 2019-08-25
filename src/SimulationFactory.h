/*
 * SimulationFactory.h
 *
 *  Created on: 26 sie 2019
 *      Author: pkua
 */

#ifndef SIMULATIONFACTORY_H_
#define SIMULATIONFACTORY_H_

#include "RandomWalker.h"

class SimulationFactory {
public:
    virtual ~SimulationFactory() = default;

    virtual RandomWalker &getRandomWalker() = 0;
};

#endif /* SIMULATIONFACTORY_H_ */
