//
// Created by Finn Lukas Busch on 9/15/22.
//

#include "scenario.h"
#include <iostream>
#include <chrono>

/**
 * Step the agents forward in real-time (because no dt is provided)
 */
template <typename a, typename s> bool scenario<a, s>::step() {
    if (last_step == 0){
        clock.restart();
    }
    sf::Time deltatime = clock.getElapsedTime();
    clock.restart();
    float dt = deltatime.asSeconds();
    last_step += dt;
    return step(dt);
}

/**
 * Steps the agents forward in real-time with provided time step.
 * @param dt time step
 */
template <typename a, typename s> bool scenario<a, s>::step(float dt) {
    if(config->viz) {
        sf::Event ev;
        keys.pressed = false;
        keys.released = false;
        keys.released_codes.clear();
        keys.pressed_codes.clear();
        while (window->pollEvent(ev)) { // Catch window event
            if(ev.type == sf::Event::KeyPressed) {
                keys.pressed = true;
                keys.pressed_codes.push_back(ev.key.code);
            }
            if(ev.type == sf::Event::KeyReleased){
                keys.released = true;
                keys.released_codes.push_back(ev.key.code);
            }
        }
    }
    return false;
}

/**
 * Add Agent
 * @tparam actuation_array actuation array type
 * @tparam state_array state array type
 * @param agent agent object to add
 */
template<typename actuation_array, typename state_array>
int scenario<actuation_array, state_array>::add_agent(rl_agent<actuation_array, state_array> *agent) {
    if(config->print_level>=2){
            std::cout << "Adding Agent of type " << agent->get_name() << std::endl;
    }
    agents.insert(std::pair<int, rl_agent<actuation_array, state_array>*>(num_agents, agent));
    return num_agents++;
}

/**
 * scenario constructor
 * @tparam actuation_array actuation array type
 * @tparam state_array state array type
 * @param config Scenario specific params
 */
template<typename actuation_array, typename state_array>
scenario<actuation_array, state_array>::scenario(GlobalParams* config) {
    this->config = config;
    if(config->viz){
        sf::ContextSettings settings;
        settings.antialiasingLevel = 16;
        window = new sf::RenderWindow(sf::VideoMode(config->resx, config->resy), "SpaceShip Sim", sf::Style::None, settings);
        view = new sf::View(sf::Vector2f(0.0f, 0.0f), sf::Vector2f(config->sizex, -config->sizey));
        window->setView(*view);
    }
    eng = new std::default_random_engine(rd());

}

/**
 * Get Scenario name
 * @tparam actuation_array
 * @tparam state_array
 * @return scenario name
 */
template<typename actuation_array, typename state_array>
std::string& scenario<actuation_array, state_array>::get_name() {
    return sim_name;
}

/**
 * Initial print of sim
 */
template<typename actuation_array, typename state_array>
void scenario<actuation_array, state_array>::print_init() {
    if(config->print_level>=2){
        std::cout << "Started Simulation " << get_name() << std::endl;
    }
}

template<typename actuation_array, typename state_array>
const KeyInput& scenario<actuation_array, state_array>::get_event() {
    return keys;
}

template<typename actuation_array, typename state_array>
scenario<actuation_array, state_array>::~scenario() {
    delete(view);
    delete(window);
    for(auto const &p: agents){
        delete(p.second);
    }

}

template class scenario<std::array<float, 4>, std::array<float, 6>>;