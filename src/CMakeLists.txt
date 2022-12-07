cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp -fPIC" )
set(CMAKE_MODULE_PATH "/usr/share/SFML/cmake/Modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED network audio graphics window system)
find_package(PythonLibs REQUIRED)
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)
FIND_PACKAGE(Boost  COMPONENTS python system numpy REQUIRED )
find_package(pybind11 CONFIG REQUIRED)
include(GNUInstallDirs)

pybind11_add_module(SpaceShipSim MODULE
        scenario.cpp rl_agent.cpp
        Controller.cpp
        SpaceShipSim/SpaceShipSim.cpp
        SpaceShipSim/SpaceShip.cpp
        SpaceShipSim/viz_helpers/emitter.cpp
        SpaceShipSim/viz_helpers/particle.cpp
        SpaceShipSim/SpaceControllers.cpp
        Label.cpp
        ThreadPool.cpp
        SpaceShipSim/viz_helpers/goal.cpp SpaceShipSim/viz_helpers/boosterglare.cpp
        SpaceShipSim/viz_helpers/boosterglare.h SpaceShipSim/viz_helpers/GlowingBall.cpp
        SpaceShipSim/viz_helpers/GlowingBall.h SpaceShipSim/DistanceSensors.cpp SpaceShipSim/DistanceSensors.h SpaceShipSim/Obstacle.cpp
        interface.cpp
        )

target_include_directories(SpaceShipSim PUBLIC ${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(SpaceShipSim PRIVATE ${PYTHON_LIBRARIES} sfml-graphics sfml-audio sfml-window sfml-system Eigen3::Eigen Python3::NumPy)

target_compile_definitions(SpaceShipSim PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS SpaceShipSim
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/PythonBuild)
