cmake_minimum_required(VERSION 3.15...3.25)
project(${SKBUILD_PROJECT_NAME}  VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
find_package (Eigen3 REQUIRED NO_MODULE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -fopenmp -fPIC" )
#set(CMAKE_MODULE_PATH "/usr/share/SFML/cmake/Modules" ${CMAKE_MODULE_PATH})
find_package(SFML 2 REQUIRED network audio graphics window system)
#find_package(PythonLibs REQUIRED)
#find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()
pybind11_add_module(pomdp_spaceship_env MODULE
        src/scenario.cpp
        src/rl_agent.cpp
        src/Controller.cpp
        src/SpaceShipSim/SpaceShipSim.cpp
        src/SpaceShipSim/SpaceShip.cpp
        src/SpaceShipSim/viz_helpers/emitter.cpp
        src/SpaceShipSim/viz_helpers/particle.cpp
        src/SpaceShipSim/SpaceControllers.cpp
        src/Label.cpp
        src/ThreadPool.cpp
        src/SpaceShipSim/viz_helpers/goal.cpp
        src/SpaceShipSim/viz_helpers/boosterglare.cpp
        src/SpaceShipSim/viz_helpers/GlowingBall.cpp
        src/SpaceShipSim/DistanceSensors.cpp
        src/SpaceShipSim/Obstacle.cpp
        src/interface.cpp
        src/resources/booster.h
        src/resources/mainframe.h
        src/resources/Arial.h
        )

target_link_libraries(pomdp_spaceship_env PRIVATE ${PYTHON_LIBRARIES} sfml-graphics sfml-audio sfml-window sfml-system Eigen3::Eigen)

target_compile_definitions(pomdp_spaceship_env PRIVATE VERSION_INFO=${PROJECT_VERSION})

install(TARGETS pomdp_spaceship_env
        DESTINATION .)

