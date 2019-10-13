/*
 * Simulation.h
 *
 *  Created on: 11 wrz 2019
 *      Author: pkua
 */

#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "MSDData.h"

/**
 * @brief The interface for a class performing a single simulation of multiple random walks of the same type, after
 * which MSDData mean square displacement data can be obtained.
 */
class Simulation {
public:
    virtual ~Simulation() = default;

    /**
     * @brief Performs the simulation.
     *
     * @param logger output stream to print info on the process
     */
    virtual void run(std::ostream &logger) = 0;

    /**
     * @brief Returns the MSDData obtained after the simulation.
     *
     * @return the MSDData obtained after the simulation
     */
    virtual MSDData &getMSDData() = 0;
};

#endif /* SIMULATION_H_ */
