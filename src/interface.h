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

PYBIND11_MODULE(pomdp_spaceship_env, m)
{

    /*m.doc() =  R"pbdoc(
        POMDP Space Ship Environment

    )pbdoc";*/
//    py::options options;
//    options.disable_function_signatures();
    py::class_<GlobalParams>(m, "Config", R"pbdoc(
    Confic datastruct.
    
    Attributes:
	    ResX (int) : Width Resolution. Defaults to ``1920``
	    ResY (int) : Height Resolution. Defaults to ``1080``
   	    SizeX (float) : Environment Width in meters. Defaults to ``170``
	    SizeY (float) : Height Resolution. Defaults to ``1080``
	    PrintLevel(int) : -/-.
	    AutoReset (bool) : A boolean determining whether a ship should be reset on terminal state. Defaults to ``True``
	    DynamicGoals (bool) : -/-. Defaults to ``False``
	    ShareEnvs (bool) : -/-. Defaults to ``False``
	    Viz (bool) : -/-. Defaults to ``True``
	    NumObs (int) : -/-. Defaults to ``0``
    )pbdoc")
    .def(py::init())
    .def_readwrite("Viz", &GlobalParams::viz)
    .def_readwrite("ResX", &GlobalParams::resx)
    .def_readwrite("ResY", &GlobalParams::resy)
    .def_readwrite("SizeX", &GlobalParams::sizex)
    .def_readwrite("SizeY", &GlobalParams::sizey)
    .def_readwrite("PrintLevel", &GlobalParams::print_level)
    .def_readwrite("AutoReset", &GlobalParams::auto_reset)
    .def_readwrite("DynamicGoals", &GlobalParams::dynamic_goal)
    .def_readwrite("ShareEnvs", &GlobalParams::share_envs)
    .def_readwrite("NumObs", &GlobalParams::num_obstacles);

    py::class_<SpaceShipInterface>(m, "Env", R"pbdoc(
        Environment class. Contains all the required methods to run and interact with the
        Environment.
        )pbdoc")
            .def(py::init<GlobalParams, int>(), R"pbdoc(
        Default Constructor for the Environment

        Args:
            config (Config) : Environment Parameters
            n_ships (int) : Number of Ships. Defaults to ``1``
        )pbdoc", py::arg("config"), py::arg("n_ships"))

        .def(py::init<GlobalParams, int, py::list>(), R"pbdoc(
        Constructor for the Environment with Ship Labels

        Args:
            config (Config) : Environment Parameters
            n_ships (int) : Number of Ships. Defaults to ``1``
            labels (list of Strings) : Labels for the Ships, used for viz.
        )pbdoc", py::arg("conf"), py::arg("n_ships"), py::arg("labels"))

        .def("Reset", &SpaceShipInterface::reset, R"pbdoc(
        Resets the entire environment. All goal points are regenerated, all ships reinitialized and all obstacles regenerated.
        )pbdoc")

        .def("Step", &SpaceShipInterface::step, R"pbdoc(
        Steps the environment. If viz was set to True, it will also draw the Environment.
        )pbdoc")

        .def("Step", &SpaceShipInterface::step_dt, R"pbdoc(
        Steps the environment given an elapsed time. If viz was set to True, it will also draw the Environment.

        Args:
            dt (float) : elapsed time.
        )pbdoc", py::arg("dt"))

        .def("GetState", &SpaceShipInterface::get_states, R"pbdoc(
        Get an array of the current states of all ships.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, state_dim)
        )pbdoc")

        .def("SetControl", &SpaceShipInterface::set_controls, R"pbdoc(
        Set the control inputs for all shapes. ``Must be of type np.float32!``

        Args:
        	ControlIn (ndarray) : Numpy Array of Shape (n_ships, control_dim) and type ``np.float32``. Providing an array of a different type yields undesired behavior.
        )pbdoc", py::arg("ControlIn"))

        .def("GetReward", &SpaceShipInterface::get_rewards, R"pbdoc(
        Get an array of the current rewards of all ships.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, 1)
        )pbdoc")

        .def("GetAgentDone", &SpaceShipInterface::get_dones, R"pbdoc(
        Get an boolean array which contains "True" if the corresponding ship has collided or met the goal condition.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, 1)
        )pbdoc")

        .def("GetMinIn", &SpaceShipInterface::get_min_in, R"pbdoc(
        Get lower actuation limit of the Environment.

        Returns:
            ndarray : Numpy Array of Shape (control_dim, 1)
        )pbdoc")

        .def("GetMaxIn", &SpaceShipInterface::get_max_in, R"pbdoc(
        Get upper actuation limit of the Environment.

        Returns:
            ndarray : Numpy Array of Shape (control_dim, 1)
        )pbdoc")

        .def("SetView", &SpaceShipInterface::set_view, R"pbdoc(
        Set the camera view (viz only).

        Args:
            width (float) : Camera view width.
            height (float) : Camera view height.
            x0 (float) : Camera centerpoint.
            y0 (float) : Camera centerpoint.
        )pbdoc", py::arg("width"), py::arg("height"), py::arg("x0"), py::arg("y0"))

        .def("Draw", &SpaceShipInterface::draw, R"pbdoc(
        Draws the Environment without Stepping the Simulation.
        )pbdoc")

        .def("SetShip", &SpaceShipInterface::set_ship, R"pbdoc(
        Set a specific ship, specified by its ID, to a specific Position.

        Args:
            id (int) : Ship Identifier
            x (float) : x coordinate
            y (float) : y coordinate
            phi (float) : angle
            vx (float) : x velocity
            vy (float) : y velocity
            vphi (float) : angular velocity
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"), py::arg("phi"), py::arg("vx"), py::arg("vy"), py::arg("vphi"))

        .def("SetGoal", &SpaceShipInterface::set_goal, R"pbdoc(
        Set a goal point, specified by the ID, to a specific position. Not working for dynamic goal points.

        Args:
            id (int) : Ship Identifier
            x (float) : x coordinate
            y (float) : y coordinate
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"))
        .def("ExportFrame", &SpaceShipInterface::export_frame, R"pbdoc(
        Export the current frame to a given path.

        Args:
            path (String) : Export path.
        )pbdoc", py::arg("path"))

        .def("SetViz", &SpaceShipInterface::set_viz, R"pbdoc(
        Disable or enable obstacle drawing and distance sensor ray drawing.
        
        Args:
        	draw_rays (bool) : Draw Rays?
        	draw_obs (bool) : Draw Obs?
        )pbdoc", py::arg("draw_rays"), py::arg("draw_obs"))

        .def("ResetToInit", &SpaceShipInterface::reset_to_init, R"pbdoc(
        Reset a specific ship, specified by its ID, to the initial position.

        Args:
            id (int) : Ship Identifier
        )pbdoc", py::arg("id"));
    m.attr("__version__") = "dev";
}

#endif //RLSIMLIBRARY_INTERFACE_H
