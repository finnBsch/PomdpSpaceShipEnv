//
// Created by finn on 9/15/22.
//

#include "rl_agent.h"
#include <iostream>

/**
 * Return state array
 * @tparam a
 * @tparam s
 * @return
 */
template <typename a, typename s> const s&  rl_agent<a, s>::get_state() const{
    return state;
}

/**
 * Initialize state and actuation to 0
 * @tparam actuation_array
 * @tparam state_array
 */
template<typename actuation_array, typename state_array>
rl_agent<actuation_array, state_array>::rl_agent(int agent_id) {
    this->agent_id = agent_id;
    for(int i = 0; i<state.size(); i++){
        state[i] = 0;
    }
    for(int i = 0; i<actuation.size(); i++){
        actuation[i] = 0;
    }
}

/**
 * Return Actuation
 * @tparam actuation_array actuation array type
 * @tparam state_array state array type
 * @return actuation
 */
template<typename actuation_array, typename state_array>
const actuation_array &rl_agent<actuation_array, state_array>::get_actuation() const {
    return actuation;
}


/**
 * Get Agent name for printing
 * @tparam actuation_array
 * @tparam state_array
 * @return agent name
 */
template<typename actuation_array, typename state_array>
std::string rl_agent<actuation_array, state_array>::get_name() {
    return agent_name;
}





template<typename actuation_array, typename state_array>
void rl_agent<actuation_array, state_array>::delta_score(float score) {
    this->score = this->score + score;
}

template<typename actuation_array, typename state_array>
void rl_agent<actuation_array, state_array>::set_id(int id) {
    agent_id = id;
}

template class rl_agent<std::array<float, 4>, std::array<float, 6>>;
