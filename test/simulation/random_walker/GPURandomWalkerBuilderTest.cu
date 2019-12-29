/*
 * CPURandomWalkerBuilderTest.cpp
 *
 *  Created on: 23 gru 2019
 *      Author: pkua
 */

#include <catch2/catch.hpp>
//#include <catch2/trompeloeil.hpp>

#include "simulation/random_walker/GPURandomWalkerBuilder.h"
#include "simulation/random_walker/GPURandomWalkerBuilder.tpp"
#include "utils/Utils.h"
//#include "../../utils/FileUtilsMocks.h"
//#include "../../image/ImageReaderMock.h"
#include "move_generator/MoveGeneratorsMocks.h"
#include "move_filter/MoveFiltersMocks.h"
#include "../../test_utils/CPUDataAccessor.h"


namespace {
    /*
     * Simple mocks which save parameters from the constructor and return default values.
     */

    struct GPURandomWalkerMock : public RandomWalker {
        std::size_t numberOfWalks;
        RandomWalker::WalkParameters walkParameters;
        std::size_t numberOfMoveFilterSetupThreads;
        MoveGenerator *moveGenerator;
        MoveFilter *moveFilter;
        std::ostream &logger;

        GPURandomWalkerMock(std::size_t numberOfWalks, WalkParameters walkParameters,
                            std::size_t numberOfMoveFilterSetupThreads, MoveGenerator *moveGenerator,
                            MoveFilter *moveFilter, std::ostream &logger)
                : numberOfWalks{numberOfWalks}, walkParameters{walkParameters},
                  numberOfMoveFilterSetupThreads{numberOfMoveFilterSetupThreads}, moveGenerator{moveGenerator},
                  moveFilter{moveFilter}, logger{logger}
        { }

        std::vector<Tracer> getRandomInitialTracersVector() override { return std::vector<Tracer>{}; }
        void run(std::ostream &logger, const std::vector<Tracer> &initialTracers) override { }
        std::size_t getNumberOfTrajectories() const override { return 0; }
        std::size_t getNumberOfSteps() const override { return 0; }
        const Trajectory &getTrajectory(std::size_t index) const override { throw std::runtime_error(""); }
    };
}

using GPURandomWalkerBuilderUnderTest = GPURandomWalkerBuilder<GPURandomWalkerMock>;

/*
 * Explicit specialization for testing with mock types.
 */
template<>
struct GPURandomWalkerBuilderTraits<GPURandomWalkerBuilderUnderTest> {
    using GaussianMoveGenerator_t = GaussianMoveGeneratorMock;
    using CauchyMoveGenerator_t = CauchyMoveGeneratorMock;
    using DefaultMoveFilter_t = DefaultMoveFilterMock;
    using ImageMoveFilterPeriodicBC_t = ImageMoveFilterPeriodicBCMock;
    using ImageMoveFilterWallBC_t = ImageMoveFilterWallBCMock;
};

using Catch::Contains;
//using trompeloeil::_;

TEST_CASE("GPURandomWalkerBuilder: basic parameters") {
    RandomWalkerFactory::WalkerParameters walkerParameters;
    walkerParameters.moveGeneratorParameters        = "GaussianMoveGenerator 3";
    walkerParameters.moveFilterParameters           = "DefaultMoveFilter";
    walkerParameters.numberOfWalksInSeries          = 10;
    walkerParameters.walkParameters.numberOfSteps   = 100;
    walkerParameters.walkParameters.tracerRadius    = 3;
    walkerParameters.walkParameters.drift           = Move{1, 2};
    walkerParameters.walkParameters.integrationStep = 0.1;
    std::ostringstream logger;

    auto walker = GPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build();
    auto walkerMock = dynamic_cast<GPURandomWalkerMock*>(walker.get());

    REQUIRE(CUDA_IS_INSTANCE_OF(walkerMock->moveGenerator, GaussianMoveGeneratorMock));
    REQUIRE(CUDA_IS_INSTANCE_OF(walkerMock->moveFilter, DefaultMoveFilterMock));
    REQUIRE(walkerMock->numberOfWalks == 10);
    REQUIRE(walkerMock->walkParameters.numberOfSteps == 100);
    REQUIRE(walkerMock->walkParameters.tracerRadius == 3);
    REQUIRE(walkerMock->walkParameters.drift == Move{1, 2});
    REQUIRE(walkerMock->walkParameters.integrationStep == Approx(0.1));
}

TEST_CASE("GPURandomWalkerBuilder: move gererator") {
    RandomWalkerFactory::WalkerParameters walkerParameters;
    walkerParameters.moveFilterParameters           = "DefaultMoveFilter";
    walkerParameters.numberOfWalksInSeries          = 10;
    walkerParameters.walkParameters.numberOfSteps   = 100;
    walkerParameters.walkParameters.tracerRadius    = 3;
    walkerParameters.walkParameters.drift           = Move{1, 2};
    walkerParameters.walkParameters.integrationStep = 0.1;
    std::ostringstream logger;

    SECTION("gaussian") {
        SECTION("correct sigma") {
            walkerParameters.moveGeneratorParameters = "GaussianMoveGenerator 3";
            auto walker = GPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build();
            auto walkerMock = dynamic_cast<GPURandomWalkerMock*>(walker.get());

            REQUIRE(CUDA_IS_INSTANCE_OF(walkerMock->moveGenerator, GaussianMoveGeneratorMock));
            auto generator = get_gpu_data_accessor<GaussianMoveGeneratorMock>(walkerMock->moveGenerator);
            REQUIRE(generator->sigma == 3);
        }

        /*SECTION("incorrect sigma") {
            SECTION("zero") {
                walkerParameters.moveGeneratorParameters = "GaussianMoveGenerator 0";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("> 0"));
            }

            SECTION("negative") {
                walkerParameters.moveGeneratorParameters = "GaussianMoveGenerator -0.3";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("> 0"));
            }
        }

        SECTION("malformed") {
            SECTION("no parameter") {
                walkerParameters.moveGeneratorParameters = "GaussianMoveGenerator";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("Malformed"));
            }

            SECTION("skrewed parameter") {
                walkerParameters.moveGeneratorParameters = "GaussianMoveGenerator killme";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("Malformed"));
            }
        }*/
    }

    /*SECTION("cauchy") {
        SECTION("correct width") {
            walkerParameters.moveGeneratorParameters = "CauchyMoveGenerator 3";
            auto walker = CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build();
            auto walkerMock = dynamic_cast<CPURandomWalkerMock*>(walker.get());

            REQUIRE(is_instance_of<CPUCauchyMoveGeneratorMock>(walkerMock->moveGenerator.get()));
            auto generator = dynamic_cast<CPUCauchyMoveGeneratorMock*>(walkerMock->moveGenerator.get());
            REQUIRE(generator->width == 3);
        }

        SECTION("incorrect width") {
            SECTION("zero") {
                walkerParameters.moveGeneratorParameters = "CauchyMoveGenerator 0";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("> 0"));
            }

            SECTION("negative") {
                walkerParameters.moveGeneratorParameters = "CauchyMoveGenerator -0.3";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("> 0"));
            }
        }

        SECTION("malformed") {
            SECTION("no parameter") {
                walkerParameters.moveGeneratorParameters = "CauchyMoveGenerator";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("Malformed"));
            }

            SECTION("skrewed parameter") {
                walkerParameters.moveGeneratorParameters = "CauchyMoveGenerator killme";

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                    Contains("Malformed"));
            }
        }
    }

    SECTION("unknown") {
        walkerParameters.moveGeneratorParameters = "KillMe 7";

        REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                            Contains("Unknown"));
    }*/
}

/*TEST_CASE("CPURandomWalkerBuilder: move filter") {
    RandomWalkerFactory::WalkerParameters walkerParameters;
    walkerParameters.moveGeneratorParameters        = "GaussianMoveGenerator 3";
    walkerParameters.numberOfWalksInSeries          = 10;
    walkerParameters.walkParameters.numberOfSteps   = 100;
    walkerParameters.walkParameters.tracerRadius    = 3;
    walkerParameters.walkParameters.drift           = Move{1, 2};
    walkerParameters.walkParameters.integrationStep = 0.1;
    std::ostringstream logger;

    SECTION("default") {
        walkerParameters.moveFilterParameters = "DefaultMoveFilter";
        auto walker = CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build();
        auto walkerMock = dynamic_cast<CPURandomWalkerMock*>(walker.get());

        REQUIRE(is_instance_of<CPUGaussianMoveGeneratorMock>(walkerMock->moveGenerator.get()));
    }

    SECTION("image") {
        auto fileIstreamProvider = std::unique_ptr<FileIstreamProviderMock>(new FileIstreamProviderMock);
        auto imageReader = std::unique_ptr<ImageReaderMock>(new ImageReaderMock);

        SECTION("image file, WallBoundaryConditions") {
            auto imageStreamPtr = new std::istringstream;
            auto imageStream = std::unique_ptr<std::istringstream>(imageStreamPtr);
            Image image(2, 3);
            image(0, 0) = 0x111111FF; image(1, 0) = 0x222222FF;
            image(0, 1) = 0x333333FF; image(1, 1) = 0x444444FF;
            image(0, 2) = 0x555555FF; image(1, 2) = 0x666666FF;
            walkerParameters.moveFilterParameters = "ImageMoveFilter image.ppm WallBoundaryConditions";
            REQUIRE_CALL_V(*fileIstreamProvider, openFile("image.ppm"),
                .LR_RETURN(std::move(imageStream)));
            REQUIRE_CALL_V(*imageReader, read(_),
                .WITH(&_1 == imageStreamPtr)
                .RETURN(image));
            auto walker = CPURandomWalkerBuilderUnderTest(1234, walkerParameters, std::move(fileIstreamProvider),
                                                          std::move(imageReader), logger).build();
            auto walkerMock = dynamic_cast<CPURandomWalkerMock*>(walker.get());

            REQUIRE(is_instance_of<ImageMoveFilterWallBCMock>(walkerMock->moveFilter.get()));
            auto filter = dynamic_cast<ImageMoveFilterWallBCMock*>(walkerMock->moveFilter.get());
            REQUIRE(filter->width == 2);
            REQUIRE(filter->height == 3);
            REQUIRE(filter->numberOfTrajectories == 10);
            auto expectedImageData = image.getIntData();
            std::vector<uint32_t> actualImageData(filter->intImageData, filter->intImageData + 6);
            REQUIRE(actualImageData == expectedImageData);
        }

        SECTION("PeriodicBoundaryConditions") {
            walkerParameters.moveFilterParameters = "ImageMoveFilter image.ppm PeriodicBoundaryConditions";
            ALLOW_CALL_V(*fileIstreamProvider, openFile("image.ppm"),
                .RETURN(nullptr));
            ALLOW_CALL_V(*imageReader, read(_),
                .RETURN(Image(1, 1)));
            auto walker = CPURandomWalkerBuilderUnderTest(1234, walkerParameters, std::move(fileIstreamProvider),
                                                          std::move(imageReader), logger).build();
            auto walkerMock = dynamic_cast<CPURandomWalkerMock*>(walker.get());

            REQUIRE(is_instance_of<ImageMoveFilterPeriodicBCMock>(walkerMock->moveFilter.get()));
        }

        SECTION("malformed") {
            SECTION("no BC") {
                walkerParameters.moveFilterParameters = "ImageMoveFilter image.ppm";
                ALLOW_CALL_V(*fileIstreamProvider, openFile("image.ppm"),
                    .RETURN(nullptr));
                ALLOW_CALL_V(*imageReader, read(_),
                    .RETURN(Image(1, 1)));

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters,
                                                                    std::move(fileIstreamProvider),
                                                                    std::move(imageReader), logger).build(),
                                    Contains("Malformed"));
            }

            SECTION("unknown BC") {
                walkerParameters.moveFilterParameters = "ImageMoveFilter image.ppm KillMe";
                ALLOW_CALL_V(*fileIstreamProvider, openFile("image.ppm"),
                    .RETURN(nullptr));
                ALLOW_CALL_V(*imageReader, read(_),
                    .RETURN(Image(1, 1)));

                REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters,
                                                                    std::move(fileIstreamProvider),
                                                                    std::move(imageReader), logger).build(),
                                    Contains("Unknown"));
            }
        }
    }

    SECTION("unknown") {
        SECTION("no parameter") {
            walkerParameters.moveFilterParameters = "KillMe";

            REQUIRE_THROWS_WITH(CPURandomWalkerBuilderUnderTest(1234, walkerParameters, logger).build(),
                                Contains("Unknown"));
        }
    }
}*/
