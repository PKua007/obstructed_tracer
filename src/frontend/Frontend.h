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
#include <type_traits>

#include "Parameters.h"
#include "Simulation.h"
#include "Analyzer.h"

template <typename ConcreteSimulation, typename ConcreteAnalyzer>
class Frontend {
    static_assert(std::is_base_of<Simulation, ConcreteSimulation>::value);
    static_assert(std::is_base_of<Analyzer, ConcreteAnalyzer>::value);

private:
    std::string command;
    std::string mode;
    Parameters parameters;
    std::vector<std::string> additionalArguments;
    std::ostream &logger;

    int perform_walk();
    int analyze();

public:
    Frontend(int argc, char **argv, std::ostream &logger);

    int run();
};

#include "Frontend.tpp"

#endif /* FRONTEND_H_ */
