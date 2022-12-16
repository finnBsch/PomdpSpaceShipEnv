//
// Created by finn on 9/18/22.
//

#include "SpaceControllers.h"
#include <iostream>
/**
 * Return controls
 * @param scen Scenario (to get states from)
 * @return controls
 */

SpaceKeyboardController::SpaceKeyboardController() {
    controller_name = "SpaceKeyboardController";
}

void SpaceKeyboardController::generate_controls(scenario<act_arr, state_arr> *scen, int agent_id) {
    auto event = scen->get_event();
    if(event.pressed){
        for(auto a:event.pressed_codes){
            switch (a) {
                case sf::Keyboard::Space:
                    fac = 2.5;
                    break;
                case sf::Keyboard::W:
                    ang += 0.1;
                    break;
                case sf::Keyboard::S:
                    ang -= 0.1;
                    break;
                case sf::Keyboard::A:
                    fac1 = 2.5;
                    break;
                case sf::Keyboard::D:
                    fac2 = 2.5;
                    break;
            }
        }
    }
    if(event.released){
        for(auto a:event.released_codes){
            switch (a) {
                case sf::Keyboard::Space:
                    fac = 0;
                    break;
                case sf::Keyboard::A:
                    fac1 = 0;
                    break;
                case sf::Keyboard::D:
                    fac2 = 0;
                    break;
            }
        }
    }
    actuation[0] = (fac1 + fac)*9.81/3;
    actuation[1] = (fac2 + fac)*9.81/3;
    actuation[2] = ang;
    actuation[3] = -ang;
}

const act_arr &SpaceKeyboardController::get_controls() {
    return actuation;
}

