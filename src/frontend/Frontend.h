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

/**
 * @brief User-visible part of the program, which, in general, reads the data, passes to the backend and stores the
 * results.
 *
 * The concrete modes (i.e. performing walks and analyzing the results) are invoked based on @a argc and @a argv
 * from the constructor.
 *
 * @tparam ConcreteSimulation the concrete implementation of Simulation to use
 * @tparam ConcreteAnalyzer the concrete implementation of Analyzer to use
 */
template <typename ConcreteSimulation, typename ConcreteAnalyzer>
class Frontend {
    static_assert(std::is_base_of<Simulation, ConcreteSimulation>::value, "");
    static_assert(std::is_base_of<Analyzer, ConcreteAnalyzer>::value, "");

private:
    std::string command;
    std::string mode;
    Parameters parameters;
    std::vector<std::string> additionalArguments;
    std::ostream &logger;

    void perform_walk();
    void analyze();

public:
    /**
     * @brief Exception during preparation, execution or saving results of one of the modes.
     */
    struct RunException : public std::runtime_error {
        RunException(const std::string &msg) : std::runtime_error(msg) { }
    };

    /**
     * @brief Creates the frontend based on main-like arguments.
     *
     * Specifically, parses the arguments and loads the parameters from file.
     *
     * @param argc like in main
     * @param argv like in main
     * @param logger @a std::ostream to log information during a whole execution
     */
    Frontend(int argc, char **argv, std::ostream &logger);

    /**
     * @brief Runs one of the modes determined in the constructor.
     *
     * Throws RunException if mode is unknown.
     */
    void run();
};

#include "Frontend.tpp"

#endif /* FRONTEND_H_ */
