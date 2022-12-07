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
#include <numpy/arrayobject.h>
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
    Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>* get_states();
    const Eigen::Array<float, Eigen::Dynamic, 1> * get_rewards();
    const Eigen::Array<bool, Eigen::Dynamic, 1> * get_dones();
    void set_controls(py::array &control_ins);
    void set_view(float width, float height, float x0, float y0);
    void draw(float dt);
    void set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi);
    void set_goal(int id, float x, float y);
    void export_frame(std::string save_path);
    void set_viz(bool draw_rays, bool draw_obs);
    void reset_to_init(int id);
};

PYBIND11_MODULE(libSpaceShipSim, m)
{
    py::class_<GlobalParams>(m, "Config")
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


    py::class_<SpaceShipInterface>(m, "SpaceShip")
            .def(py::init<GlobalParams, int>())
            .def(py::init<GlobalParams, int, py::list>())
            .def("Reset", &SpaceShipInterface::reset)
            .def("Step", &SpaceShipInterface::step)
            .def("Step", &SpaceShipInterface::step_dt)
            .def("GetState", &SpaceShipInterface::get_states)
            .def("SetControl", &SpaceShipInterface::set_controls)
            .def("GetReward", &SpaceShipInterface::get_rewards)
            .def("GetAgentDone", &SpaceShipInterface::get_dones)
            .def("GetMinIn", &SpaceShipInterface::get_min_in)
            .def("GetMaxIn", &SpaceShipInterface::get_max_in)
            .def("SetView", &SpaceShipInterface::set_view)
            .def("Draw", &SpaceShipInterface::draw)
            .def("SetShip", &SpaceShipInterface::set_ship)
            .def("SetGoal", &SpaceShipInterface::set_goal)
            .def("ExportFrame", &SpaceShipInterface::export_frame)
            .def("SetViz", &SpaceShipInterface::set_viz)
            .def("ResetToInit", &SpaceShipInterface::reset_to_init);
}

#endif //RLSIMLIBRARY_INTERFACE_H
