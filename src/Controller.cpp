//
// Created by finn on 9/15/22.
//

#include "Controller.h"
#include <array>
#include "scenario.h"
#include <iostream>
/**
 * Update and return controls
 * @tparam actuation_array template for actuation array
 * @param scen scenario pointer to get world state from.
 */

/**
 * Get controls function. Defaults to no update
 * @tparam actuation_array actuation array type
 * @tparam state_array state array type
 * @param scen Scenario. To get world state
 * @return actuation
 */
template<typename actuation_array, typename state_array>
void Controller<actuation_array, state_array>::generate_controls(scenario<actuation_array, state_array> *scen, int agent_id) {

}

/**
 * Default constructor. Initializes actuation to 0
 * @tparam actuation_array actuation array type
 * @tparam state_array state array type
 */
template<typename actuation_array, typename state_array>
Controller<actuation_array, state_array>::Controller() {
    for(int i = 0; i < actuation.size(); i++){
        actuation[i] = 0;
    }
}

template<typename actuation_array, typename state_array>
std::string Controller<actuation_array, state_array>::get_name() {
    return controller_name;
}


template<typename actuation_array, typename state_array>
const actuation_array &Controller<actuation_array, state_array>::get_controls() {
    return actuation;
}
template<typename actuation_array, typename state_array>
void Controller<actuation_array, state_array>::set_controls(actuation_array controls) {
    Controller<actuation_array, state_array>::actuation = controls;
}

template<typename actuation_array, typename state_array>
const std::vector<float> &Controller<actuation_array, state_array>::get_state() {
    return Controller<actuation_array, state_array>::world_state;
}

template<typename actuation_array, typename state_array>
void Controller<actuation_array, state_array>::reset() {
    for(int i = 0; i < actuation.size(); i++){
        actuation[i] = 0;
    }
}

/**
 * External Controller
 */
template<typename actuation_array, typename state_array>
ExternalController<actuation_array, state_array>::ExternalController() {
        Controller<actuation_array, state_array>::controller_name = "ExternalController";
}

template<typename actuation_array, typename state_array>
void ExternalController<actuation_array, state_array>::generate_controls(scenario<actuation_array, state_array> *scen,
                                                                         int agent_id) {
//    Controller<actuation_array, state_array>::world_state = scen->get_relative_state(agent_id);
}


template class Controller<std::array<float, 4>, std::array<float, 6>>;
template class ExternalController<std::array<float, 4>, std::array<float, 6>>;