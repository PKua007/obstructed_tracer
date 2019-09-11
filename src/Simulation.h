/*
 * Simulation.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "MSDData.h"

class Simulation {
public:
    virtual ~Simulation() = default;

    virtual void run(std::ostream &logger) = 0;
    virtual MSDData &getMSDData() = 0;
};

#endif /* SIMULATION_H_ */
