//
// Created by finn on 9/19/22.
//

#ifndef RLSIMLIBRARY_INTERFACE_H
#define RLSIMLIBRARY_INTERFACE_H

#include "SpaceShipSim/SpaceShipSim.h"
#include "SpaceShipSim/SpaceControllers.h"
#include "scenario.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;


class SpaceShipInterface{
private:
    SpaceShipSim* sim;
    GlobalParams config;
    Eigen::Array<float, Eigen::Dynamic, 4> control_in;
    int n_ships = 1;
    std::vector<std::string> labels;
public:
    SpaceShipInterface(GlobalParams config, int n_ships);
    SpaceShipInterface(GlobalParams config, int n_ships, py::list ns);
    ~SpaceShipInterface();
    bool step();
    bool step_dt(float dt);
    void reset();
    const act_arr* get_max_in();
    const act_arr* get_min_in();
    Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>& get_states();
    const Eigen::Array<float, Eigen::Dynamic, 1>& get_rewards();
    const Eigen::Array<bool, Eigen::Dynamic, 1>& get_dones();
    void set_controls(py::array &control_ins);
    void set_view(float width, float height, float x0, float y0);
    void draw(float dt);
    void set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi);
    void set_goal(int id, float x, float y);
    void export_frame(std::string save_path);
    void set_viz(bool draw_rays, bool draw_obs);
    void reset_to_init(int id);
};

PYBIND11_MODULE(env_core, m)
{

    m.doc() =  R"pbdoc(
        POMDP Space Ship Environment
        -----------------------
    )pbdoc";
//    py::options options;
//    options.disable_function_signatures();
    py::class_<GlobalParams>(m, "Config")
            .def(py::init())
            .def_readwrite("Viz", &GlobalParams::viz, R"pbdoc(
                Visualize the environment.
                Type: bool
                Default: False
            )pbdoc")
            .def_readwrite("ResX", &GlobalParams::resx, R"pbdoc(
    Resolution in width.
            Type: int
    Default: 1920
    )pbdoc")
            .def_readwrite("ResY", &GlobalParams::resy, R"pbdoc(
    Resolution in height.
            Type: int
    Default: 1080
    )pbdoc")
            .def_readwrite("SizeX", &GlobalParams::sizex, R"pbdoc(
    Environment size width
            Type: float
    Default: 170
    )pbdoc")
            .def_readwrite("SizeY", &GlobalParams::sizey, R"pbdoc(
    Environment size height
            Type: float
    Default: 100
    )pbdoc")
            .def_readwrite("PrintLevel", &GlobalParams::print_level, R"pbdoc(
    Output print Level
            Type: int
    Default: 0
    )pbdoc")
            .def_readwrite("AutoReset", &GlobalParams::auto_reset, R"pbdoc(
    Determines whether one Space Ship should be reset upon collision or goal condition
            Type: bool
    Default: True
    )pbdoc")
            .def_readwrite("DynamicGoals", &GlobalParams::dynamic_goal, R"pbdoc(
    Determines whether the goal points should be moving (dynamic) or not (static)
            Type: bool
    Default: False
    )pbdoc")
            .def_readwrite("ShareEnvs", &GlobalParams::share_envs,
                           R"pbdoc(
    Determines whether the Space Ships should share the Goal Point and Obstacles
            Type: bool
    Default: False
    )pbdoc")
            .def_readwrite("NumObs", &GlobalParams::num_obstacles,
                           R"pbdoc(
    Number of obstacles to be generated per Space Ship
            Type: int
    Default: 0
    )pbdoc");


    py::class_<SpaceShipInterface>(m, "Env")
            .def(py::init<GlobalParams, int>(), R"pbdoc(
        Default Constructor for the Environment
        Inputs:
            Environment Parameters, config: Config
            Number of Ships, n_ships: int
        )pbdoc", py::arg("config"), py::arg("n_ships"))
            .def(py::init<GlobalParams, int, py::list>(), R"pbdoc(
        Constructor for the Environment with Ship Labels
        Inputs:
            Environment Parameters, config: Config
            Number of Ships, n_ships: int
            Labels, labels: List of String
        )pbdoc", py::arg("conf"), py::arg("n_ships"), py::arg("labels"))
            .def("Reset", &SpaceShipInterface::reset, R"pbdoc(
        Resets the entire environment. All goal points are regenerated, all ships reinitialized and all obstacles resetted.
        )pbdoc")
            .def("Step", &SpaceShipInterface::step, R"pbdoc(
        Steps the environment. If viz was set to True, it will also draw the Environment.
        )pbdoc")
            .def("Step", &SpaceShipInterface::step_dt, R"pbdoc(
        Steps the environment given an elapsed time. If viz was set to True, it will also draw the Environment.
        )pbdoc", py::arg("dt"))
            .def("GetState", &SpaceShipInterface::get_states, R"pbdoc(
        Get an array of the current states of all ships.
        Returns:
            Numpy Array of Shape (n_ships, state_dim)
        )pbdoc")
            .def("SetControl", &SpaceShipInterface::set_controls, R"pbdoc(
        Set the control inputs for all shapes. MUST BE OF TYPE np.float32!
        Input:
            Numpy Array of Shape (n_ships, control_dim)
        )pbdoc", py::arg("ControlIn"))
            .def("GetReward", &SpaceShipInterface::get_rewards, R"pbdoc(
        Get an array of the current rewards of all ships.

        Returns
        -------
        out : ndarray
            An NumPy Array of Shape (n_ships, 1)


        )pbdoc")
            .def("GetAgentDone", &SpaceShipInterface::get_dones, R"pbdoc(
        Get an boolean array which contains "True" if the corresponding ship has collided or met the goal condition.
        Returns:
            Numpy Array of Shape (n_ships, 1)
        )pbdoc")
            .def("GetMinIn", &SpaceShipInterface::get_min_in, R"pbdoc(
        Get lower actuation limit of the Environment.
        Returns:
            Numpy Array of Shape (control_dim, 1)
        )pbdoc")
            .def("GetMaxIn", &SpaceShipInterface::get_max_in, R"pbdoc(
        Get upper actuation limit of the Environment.
        Returns:
            Numpy Array of Shape (control_dim, 1)
        )pbdoc")
            .def("SetView", &SpaceShipInterface::set_view, R"pbdoc(
        Set the camera view (viz only).
        Input:
            width: float
            height: float
            x0: float
            y0: float
        )pbdoc", py::arg("width"), py::arg("height"), py::arg("x0"), py::arg("y0"))
            .def("Draw", &SpaceShipInterface::draw, R"pbdoc(
        Draws the Environment without Stepping the Simulation.
        )pbdoc")
            .def("SetShip", &SpaceShipInterface::set_ship, R"pbdoc(
        Set a specific ship, specified by its ID, to a specific Position.
        Input:
            Ship Identifier, id: int
            x-coordinate, x: float
            y-coordinate, y: float
            angle, phi: float
            x-velocity, vx: float
            y-velocity, vy: float
            angular-velocity, vphi: float
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"), py::arg("phi"), py::arg("vx"), py::arg("vy"), py::arg("vphi"))
            .def("SetGoal", &SpaceShipInterface::set_goal, R"pbdoc(
        Set a goal point, specified by the ID, to a specific position. Not working for dynamic goal points.
        Input:
            Ship Identifier, id: int
            x-coordinate, x: float
            y-coordinate, y: float
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"))
            .def("ExportFrame", &SpaceShipInterface::export_frame)
            .def("SetViz", &SpaceShipInterface::set_viz, R"pbdoc(
        Disable or enable obstacle drawing and distance sensor ray drawing.
        Input:
            draw_rays: bool
            draw_obs: bool
        )pbdoc", py::arg("draw_rays"), py::arg("draw_obs"))
            .def("ResetToInit", &SpaceShipInterface::reset_to_init, R"pbdoc(
        Reset a specific ship, specified by its ID, to the initial position.
        Input:
            id: int
        )pbdoc", py::arg("id"));
    m.attr("__version__") = "dev";
}

#endif //RLSIMLIBRARY_INTERFACE_H
