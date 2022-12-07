//
// Created by finn on 9/19/22.
//

#include "interface.h"
#include "rl_agent.h"
#include "Controller.h"


/**
 * Step sim
 */
bool SpaceShipInterface::step() {
    return sim->step();
}

bool SpaceShipInterface::step_dt(float dt) {
    return sim->step(dt);
}

SpaceShipInterface::SpaceShipInterface(GlobalParams config, int n_ships, py::list ns){
    Py_Initialize();
    this->n_ships = n_ships;
    control_in.resize(n_ships, 4);
    this->config = config;
    for(int i = 0; i < len(ns); i++){
        labels.push_back(py::cast<std::string>(ns[i]));
    }
    sim = new SpaceShipSim(&this->config, n_ships, labels);
}

/**
 * Create sim.
 * @param viz boolean for viz or no viz
 * @param print_level print level. see scenario.h for def.
 */
SpaceShipInterface::SpaceShipInterface(GlobalParams config, int n_ships){
    Py_Initialize();
    this->n_ships = n_ships;
    this->config = config;

    std::cout << config.print_level << " " << std::endl;
    control_in.resize(n_ships, 4);
    sim = new SpaceShipSim(&this->config, n_ships);
}

Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>* SpaceShipInterface::get_states() {
    return sim->get_relative_state_();
}

const Eigen::Array<bool, Eigen::Dynamic, 1>*  SpaceShipInterface::get_dones(){
    return sim->get_dones();
}

const Eigen::Array<float, Eigen::Dynamic, 1> * SpaceShipInterface::get_rewards(){
    return sim->get_rewards();
}
const act_arr* SpaceShipInterface::get_min_in(){
    return sim->get_min_in();
}

const act_arr* SpaceShipInterface::get_max_in(){
    return sim->get_max_in();
}
void SpaceShipInterface::set_controls(py::array &control_ins) {

    control_in = Eigen::Map<Eigen::Array<float, 4, Eigen::Dynamic>, Eigen::Unaligned, Eigen::OuterStride<>>
    ((float *)  control_ins.data(),4, control_ins.shape(0), Eigen::OuterStride<>(4)).transpose();
    sim->set_controls(control_in);
}


void SpaceShipInterface::reset() {
    sim->reset();
}


SpaceShipInterface::~SpaceShipInterface() {
    delete(sim);
}

void SpaceShipInterface::set_view(float width, float height, float x0, float y0) {
    sim->set_view(width, height, x0, y0);
}

void SpaceShipInterface::draw(float dt) {
    sim->draw(dt);
}

void SpaceShipInterface::set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi) {
    sim->set_ship(id, x, y, phi, vx, vy, vphi);
}

void SpaceShipInterface::set_goal(int id, float x, float y) {
    sim->set_goal(id, x, y);
}
void SpaceShipInterface::export_frame(std::string save_path){
    sim->save_frame(save_path);
}

void SpaceShipInterface::set_viz(bool draw_rays, bool draw_obs) {
    sim->set_viz(draw_rays, draw_obs);
}

void SpaceShipInterface::reset_to_init(int id) {
    sim->reset_ship_to_init(id);
}
