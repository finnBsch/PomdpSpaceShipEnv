//
// Created by finn on 9/18/22.
//

#ifndef RLSIMLIBRARY_SPACECONTROLLERS_H
#define RLSIMLIBRARY_SPACECONTROLLERS_H

#include "../Controller.h"
#include "SpaceShip.h"

/**
 * Scenario specific Controllers
 */
class SpaceKeyboardController: public KeyboardController<act_arr, state_arr>{
private:
    float ang = 0;
    float fac1 = 0;
    float fac2 = 0;
    float fac = 0;

public:
    void generate_controls(scenario<act_arr , state_arr>* scen, int agent_id) override;
    const act_arr & get_controls() override;
    SpaceKeyboardController();
};

#endif //RLSIMLIBRARY_SPACECONTROLLERS_H
