/*
 * SimulationTest.cpp
 *
 *  Created on: 21 gru 2019
 *      Author: pkua
 */


#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "../frontend/MSDDataApproxEqualMatcher.h"
#include "simulation/SimulationImpl.h"
#include "RandomWalkerMock.h"
#include "RandomWalkerFactoryMock.h"

using Catch::Contains;

namespace {
    class TrajectoryPrinterMock : public SimulationImpl::TrajectoryPrinter {
    public:
        MAKE_MOCK2(print, void (const Trajectory &, const std::string &), override);
    };

    Parameters get_default_parameters() {
        Parameters params;

        params.numberOfSteps         = 1000;
        params.tracerRadius          = 0.f;
        params.moveGenerator         = "GaussianMoveGenerator";
        params.moveFilter            = "DefaultMoveFilter";
        params.integrationStep       = 1;
        params.drift                 = "xy 0 0";
        params.numberOfWalksInSeries = 1;
        params.numberOfSplits        = 1;
        params.numberOfSeries        = 1;
        params.storeTrajectories     = false;
        params.seed                  = "random";
        params.device                = "cpu";

        return params;
    }
}

TEST_CASE("Simulation: parsing basic parameters") {
    SECTION("basic") {
        auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);

        Parameters parameters = get_default_parameters();
        parameters.numberOfSteps = 2000;
        parameters.tracerRadius = 1;
        parameters.integrationStep = 2;
        parameters.numberOfWalksInSeries = 5;
        parameters.moveGenerator = "i am the MoveGenerator";

        std::ostringstream logger;
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


        auto actual = simulation.getWalkerParametersForSimulation(0);


        REQUIRE(actual.numberOfWalksInSeries == 5);
        REQUIRE(actual.walkParameters.numberOfSteps == 2000);
        REQUIRE(actual.walkParameters.tracerRadius == 1);
        REQUIRE(actual.walkParameters.integrationStep == 2);
        REQUIRE(actual.moveGeneratorParameters == "i am the MoveGenerator");
    }
}

TEST_CASE("Simulation: parsing drift") {
    auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
    auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
    std::ostringstream logger;
    Parameters parameters = get_default_parameters();

    SECTION("xy") {
        parameters.drift = "xy 5 6";
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        auto actual = simulation.getWalkerParametersForSimulation(0);

        REQUIRE(actual.walkParameters.drift == Move{5, 6});
    }

    SECTION("rt") {
        parameters.drift = "rt 8 0";
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                              logger);

        auto actual = simulation.getWalkerParametersForSimulation(0);

        REQUIRE(actual.walkParameters.drift == Move{8, 0});
    }

    SECTION("none") {
        parameters.drift = "killme 1 2";

        REQUIRE_THROWS_WITH(
            SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
            Contains("xy") && Contains("rt")
        );
    }

    SECTION("wrong xy") {
        SECTION("zero arguments") {
            parameters.drift = "xy";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("xy")
            );
        }

        SECTION("one argument") {
            parameters.drift = "xy 1";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("xy")
            );
        }

        SECTION("corrupted argument") {
            parameters.drift = "xy 1 a";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("xy")
            );
        }
    }

    SECTION("wrong rt") {
        SECTION("zero arguments") {
            parameters.drift = "rt";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("rt")
            );
        }

        SECTION("one argument") {
            parameters.drift = "rt 1";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("rt")
            );
        }

        SECTION("corrupted argument") {
            parameters.drift = "rt 1 a";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("rt")
            );
        }

        SECTION("negative radius") {
            parameters.drift = "rt -6 1";

            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("r >= 0")
            );
        }
    }
}

TEST_CASE("Simulation: parsing MoveFilter-s") {
    auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
    auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
    std::ostringstream logger;
    Parameters parameters = get_default_parameters();

    SECTION("single") {
        parameters.moveFilter = "single";
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        REQUIRE(simulation.getNumberOfSimulations() == 1);
        REQUIRE(simulation.getWalkerParametersForSimulation(0).moveFilterParameters == "single");
    }

    SECTION("couple of") {
        parameters.moveFilter = "one ; two ; three";
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        REQUIRE(simulation.getNumberOfSimulations() == 3);
        REQUIRE(simulation.getWalkerParametersForSimulation(0).moveFilterParameters == "one");
        REQUIRE(simulation.getWalkerParametersForSimulation(1).moveFilterParameters == "two");
        REQUIRE(simulation.getWalkerParametersForSimulation(2).moveFilterParameters == "three");
    }

    SECTION("messed syntax") {
        parameters.moveFilter = " ;   one ;;  ; two ; ; ;  three ;;;; ";
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        REQUIRE(simulation.getNumberOfSimulations() == 3);
        REQUIRE(simulation.getWalkerParametersForSimulation(0).moveFilterParameters == "one");
        REQUIRE(simulation.getWalkerParametersForSimulation(1).moveFilterParameters == "two");
        REQUIRE(simulation.getWalkerParametersForSimulation(2).moveFilterParameters == "three");
    }
}

TEST_CASE("Simulation: parsing device") {
    using trompeloeil::_;
    auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
    auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
    Parameters parameters = get_default_parameters();
    std::ostringstream logger;

    SECTION("CPU") {
        parameters.device = "cpu";
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .THROW(std::runtime_error("RandomWalker on CPU has been summoned")));
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        REQUIRE_THROWS_WITH(simulation.run(logger), Contains("RandomWalker on CPU has been summoned"));
    }

    SECTION("GPU") {
        parameters.device = "gpu";
        REQUIRE_CALL_V(*randomWalkerFactory, createGPURandomWalker(_, _),
            .THROW(std::runtime_error("RandomWalker on GPU has been summoned")));
        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "",
                                  logger);

        REQUIRE_THROWS_WITH(simulation.run(logger), Contains("RandomWalker on GPU has been summoned"));
    }

    SECTION("wrong GPU heap size") {
        SECTION("zero") {
            parameters.device = "gpu 0";
            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("> 0")
            );
        }

        SECTION("negative") {
            parameters.device = "gpu -5";
            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("> 0")
            );
        }

        SECTION("corrupted") {
            parameters.device = "gpu killme";
            REQUIRE_THROWS_WITH(
                SimulationImpl(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger),
                Contains("gpu")
            );
        }
    }
}

TEST_CASE("Simulation: handling split") {
    using trompeloeil::_;
    auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
    auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
    auto cpuRandomWalkerPtr = new RandomWalkerMock;
    auto cpuRandomWalker = std::unique_ptr<RandomWalkerMock>(cpuRandomWalkerPtr);
    std::ostringstream logger;
    Parameters parameters = get_default_parameters();
    parameters.device = "cpu";
    parameters.numberOfSteps = 2000;
    parameters.numberOfSplits = 2;

    REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
        .WITH(_2.walkParameters.numberOfSteps == 1000)
        .LR_RETURN(std::move(cpuRandomWalker)));
    REQUIRE_CALL_V(*randomWalkerFactory, createSplitRandomWalker(2, _),
        .WITH(_2.get() == cpuRandomWalkerPtr)
        .THROW("SplitRandomWalker has been summoned"));

    SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


    REQUIRE_THROWS_WITH(simulation.run(logger), Contains("SplitRandomWalker has been summoned"));
}

TEST_CASE("Simulation: run") {
    using trompeloeil::_;
    auto trajectoryPrinter = std::unique_ptr<TrajectoryPrinterMock>(new TrajectoryPrinterMock);
    std::ostringstream logger;

    // All but numberOfSeries, numberOfWalksInSeries and numberOfSteps, which are set in indivual sections
    Parameters parameters = get_default_parameters();
    parameters.device                = "cpu";
    parameters.moveGenerator         = "GaussianMoveGenerator";
    parameters.moveFilter            = "DefaultMoveFilter";
    parameters.drift                 = "xy 1 2";
    parameters.tracerRadius          = 3;
    parameters.integrationStep       = 2;
    parameters.numberOfSteps         = 1;

    SECTION("passing parameters") {
        parameters.numberOfSeries        = 7;
        parameters.numberOfWalksInSeries = 8;
        parameters.numberOfSteps         = 9;

        RandomWalkerFactory::WalkerParameters expectedWalkerParameters;
        expectedWalkerParameters.moveGeneratorParameters        = parameters.moveGenerator;
        expectedWalkerParameters.moveFilterParameters           = parameters.moveFilter;
        expectedWalkerParameters.numberOfWalksInSeries          = parameters.numberOfWalksInSeries;
        expectedWalkerParameters.walkParameters.numberOfSteps   = parameters.numberOfSteps;
        expectedWalkerParameters.walkParameters.tracerRadius    = parameters.tracerRadius;
        expectedWalkerParameters.walkParameters.drift           = Move{1, 2};
        expectedWalkerParameters.walkParameters.integrationStep = parameters.integrationStep;

        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, expectedWalkerParameters),
            .THROW(std::runtime_error("Got right parameters!")));

        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


        REQUIRE_THROWS_WITH(simulation.run(logger), Contains("Got right parameters!"));
    }

    // All runs will produce the same trajectories and MSDData
    Tracer initialTracer1{Point{1, 2}, 3};
    Tracer initialTracer2{Point{3, 4}, 3};
    Tracer initialTracer3{Point{9, 10}, 3};
    Tracer initialTracer4{Point{13, 14}, 3};

    Trajectory t1, t2, t3, t4;
    t1.addPoint(Point{1, 2}, false);
    t1.addPoint(Point{3, 6}, false);
    t2.addPoint(Point{1, 2}, false);
    t2.addPoint(Point{5, 8}, false);
    t3.addPoint(Point{9, 10}, false);
    t3.addPoint(Point{11, 12}, false);
    t4.addPoint(Point{13, 14}, false);
    t4.addPoint(Point{15, 16}, false);

    MSDData expectedMsd;
    std::stringstream expectedMsdStream;
    expectedMsdStream << "0 0 0 0 0" << std::endl << "3 5 10 26 16" << std::endl;
    expectedMsdStream >> expectedMsd;

    SECTION("1 simulation, 1 series, 2 walks") {
        parameters.storeTrajectories     = false;
        parameters.numberOfSeries        = 1;
        parameters.numberOfWalksInSeries = 2;

        std::vector<Tracer> initialTracers{initialTracer1, initialTracer2};

        auto randomWalker = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        REQUIRE_CALL_V(*randomWalker, getRandomInitialTracersVector(),
            .RETURN(initialTracers));
        ALLOW_CALL_V(*randomWalker, getNumberOfTrajectories(),
            .RETURN(2));
        ALLOW_CALL_V(*randomWalker, getNumberOfSteps(),
            .RETURN(1));
        REQUIRE_CALL_V(*randomWalker, run(_, initialTracers));
        REQUIRE_CALL_V(*randomWalker, getTrajectory(0),
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t1)));
        REQUIRE_CALL_V(*randomWalker, getTrajectory(1),
            .TIMES(AT_LEAST(1))
            .LR_RETURN(t2));

        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .LR_RETURN(std::move(randomWalker)));

        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


        simulation.run(logger);
        auto msd = simulation.getMSDData();


        REQUIRE_THAT(msd, IsApproxEqual(expectedMsd, 1e-8));
    }

    SECTION("1 simulation, 2 series, 1 walk") {
        parameters.storeTrajectories     = false;
        parameters.numberOfSeries        = 2;
        parameters.numberOfWalksInSeries = 1;

        std::vector<Tracer> initialTracers1{initialTracer1};
        std::vector<Tracer> initialTracers2{initialTracer2};

        auto randomWalker = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        ALLOW_CALL_V(*randomWalker, getNumberOfTrajectories(),
            .RETURN(1));
        ALLOW_CALL_V(*randomWalker, getNumberOfSteps(),
            .RETURN(1));

        trompeloeil::sequence seq;

        // First series
        REQUIRE_CALL_V(*randomWalker, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seq)
            .RETURN(initialTracers1));
        REQUIRE_CALL_V(*randomWalker, run(_, initialTracers1),
            .IN_SEQUENCE(seq));
        REQUIRE_CALL_V(*randomWalker, getTrajectory(0),
            .IN_SEQUENCE(seq)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t1)));

        // Second series
        REQUIRE_CALL_V(*randomWalker, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seq)
            .RETURN(initialTracers2));
        REQUIRE_CALL_V(*randomWalker, run(_, initialTracers2),
            .IN_SEQUENCE(seq));
        REQUIRE_CALL_V(*randomWalker, getTrajectory(0),
            .IN_SEQUENCE(seq)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t2)));

        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .LR_RETURN(std::move(randomWalker)));

        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


        simulation.run(logger);
        auto msd = simulation.getMSDData();


        REQUIRE_THAT(msd, IsApproxEqual(expectedMsd, 1e-8));
    }

    SECTION("2 simulation, 1 series, 1 walks") {
        parameters.storeTrajectories     = false;
        parameters.numberOfSeries        = 1;
        parameters.numberOfWalksInSeries = 2;
        parameters.moveFilter            = "DefaultMoveFilter ; DefaultMoveFilter";

        std::vector<Tracer> initialTracers1{initialTracer1};
        std::vector<Tracer> initialTracers2{initialTracer2};

        auto randomWalker1 = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        REQUIRE_CALL_V(*randomWalker1, getRandomInitialTracersVector(),
            .RETURN(initialTracers1));
        ALLOW_CALL_V(*randomWalker1, getNumberOfTrajectories(),
            .RETURN(1));
        ALLOW_CALL_V(*randomWalker1, getNumberOfSteps(),
            .RETURN(1));
        REQUIRE_CALL_V(*randomWalker1, run(_, initialTracers1));
        REQUIRE_CALL_V(*randomWalker1, getTrajectory(0),
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t1)));

        auto randomWalker2 = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        REQUIRE_CALL_V(*randomWalker2, getRandomInitialTracersVector(),
            .RETURN(initialTracers2));
        ALLOW_CALL_V(*randomWalker2, getNumberOfTrajectories(),
            .RETURN(1));
        ALLOW_CALL_V(*randomWalker2, getNumberOfSteps(),
            .RETURN(1));
        REQUIRE_CALL_V(*randomWalker2, run(_, initialTracers2));
        REQUIRE_CALL_V(*randomWalker2, getTrajectory(0),
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t2)));

        trompeloeil::sequence seq;

        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .IN_SEQUENCE(seq)
            .LR_RETURN(std::move(randomWalker1)));
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .IN_SEQUENCE(seq)
            .LR_RETURN(std::move(randomWalker2)));

        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter), "", logger);


        simulation.run(logger);
        auto msd = simulation.getMSDData();


        REQUIRE_THAT(msd, IsApproxEqual(expectedMsd, 1e-8));
    }

    SECTION("storing trajectories - 2 simulation, 2 series, 1 trajectory") {
        parameters.storeTrajectories     = true;
        parameters.numberOfSeries        = 2;
        parameters.numberOfWalksInSeries = 1;
        parameters.moveFilter            = "DefaultMoveFilter ; DefaultMoveFilter";

        std::vector<Tracer> initialTracers1{initialTracer1};
        std::vector<Tracer> initialTracers2{initialTracer2};
        std::vector<Tracer> initialTracers3{initialTracer3};
        std::vector<Tracer> initialTracers4{initialTracer4};

        // First simulation - RandomWalker
        auto randomWalker1 = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        ALLOW_CALL_V(*randomWalker1, getNumberOfTrajectories(),
            .RETURN(1));
        ALLOW_CALL_V(*randomWalker1, getNumberOfSteps(),
            .RETURN(1));

        trompeloeil::sequence seqRW1;

        // First simlation, first series - trajectory t1 = 0_0
        REQUIRE_CALL_V(*randomWalker1, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seqRW1)
            .RETURN(initialTracers1));
        REQUIRE_CALL_V(*randomWalker1, run(_, initialTracers1),
            .IN_SEQUENCE(seqRW1));
        REQUIRE_CALL_V(*randomWalker1, getTrajectory(0),
            .IN_SEQUENCE(seqRW1)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t1)));

        // First simulation, second series - trajectory t2 = 0_1
        REQUIRE_CALL_V(*randomWalker1, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seqRW1)
            .RETURN(initialTracers2));
        REQUIRE_CALL_V(*randomWalker1, run(_, initialTracers2),
            .IN_SEQUENCE(seqRW1));
        REQUIRE_CALL_V(*randomWalker1, getTrajectory(0),
            .IN_SEQUENCE(seqRW1)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t2)));

        // Second simulation - RandomWalker - trajectory t3 = 1_0
        auto randomWalker2 = std::unique_ptr<RandomWalkerMock>(new RandomWalkerMock);
        ALLOW_CALL_V(*randomWalker2, getNumberOfTrajectories(),
            .RETURN(1));
        ALLOW_CALL_V(*randomWalker2, getNumberOfSteps(),
            .RETURN(1));

        trompeloeil::sequence seqRW2;

        // Second simlation, first series - trajectory t4 = 1_1
        REQUIRE_CALL_V(*randomWalker2, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seqRW2)
            .RETURN(initialTracers3));
        REQUIRE_CALL_V(*randomWalker2, run(_, initialTracers3),
            .IN_SEQUENCE(seqRW2));
        REQUIRE_CALL_V(*randomWalker2, getTrajectory(0),
            .IN_SEQUENCE(seqRW2)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t3)));

        // Second simulation, second series
        REQUIRE_CALL_V(*randomWalker2, getRandomInitialTracersVector(),
            .IN_SEQUENCE(seqRW2)
            .RETURN(initialTracers4));
        REQUIRE_CALL_V(*randomWalker2, run(_, initialTracers4),
            .IN_SEQUENCE(seqRW2));
        REQUIRE_CALL_V(*randomWalker2, getTrajectory(0),
            .IN_SEQUENCE(seqRW2)
            .TIMES(AT_LEAST(1))
            .LR_RETURN((t4)));

        auto randomWalkerFactory = std::unique_ptr<RandomWalkerFactoryMock>(new RandomWalkerFactoryMock);
        trompeloeil::sequence seqRWF;
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .IN_SEQUENCE(seqRWF)
            .LR_RETURN(std::move(randomWalker1)));
        REQUIRE_CALL_V(*randomWalkerFactory, createCPURandomWalker(_, _),
            .IN_SEQUENCE(seqRWF)
            .LR_RETURN(std::move(randomWalker2)));

        // We want correct names for correct trajectories
        REQUIRE_CALL_V(*trajectoryPrinter, print(_, "trajectory_0_0.txt"),
            .LR_WITH(&_1 == &t1));
        REQUIRE_CALL_V(*trajectoryPrinter, print(_, "trajectory_0_1.txt"),
            .LR_WITH(&_1 == &t2));
        REQUIRE_CALL_V(*trajectoryPrinter, print(_, "trajectory_1_0.txt"),
            .LR_WITH(&_1 == &t3));
        REQUIRE_CALL_V(*trajectoryPrinter, print(_, "trajectory_1_1.txt"),
            .LR_WITH(&_1 == &t4));

        SimulationImpl simulation(parameters, std::move(randomWalkerFactory), std::move(trajectoryPrinter),
                                  "trajectory", logger);


        simulation.run(logger);
    }
}
