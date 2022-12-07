//
// Created by finn on 9/15/22.
//

#ifndef RLSIMLIBRARY_CONTROLLER_H
#define RLSIMLIBRARY_CONTROLLER_H
#include <vector>
#include <string>
//#include "scenario.h"

template<typename a, typename s> class scenario;


template <typename actuation_array, typename state_array> class Controller {
private:
protected:
    actuation_array actuation;
    std::vector<float> world_state;
    std::string controller_name;
public:
    Controller();
    void reset();
    virtual void generate_controls(scenario<actuation_array, state_array>* scen, int agent_id);
    const std::vector<float>& get_state();
    void set_controls(actuation_array controls);
    virtual const actuation_array& get_controls();
    std::string get_name();
};

template <typename actuation_array, typename state_array> class ExternalController: public Controller<actuation_array, state_array>{
private:
protected:
public:
    void generate_controls(scenario<actuation_array, state_array>* scen, int agent_id) override;
    ExternalController();
};

template<typename actuation_array, typename state_array> class KeyboardController: public Controller<actuation_array, state_array>{
private:
protected:
public:
};



#endif //RLSIMLIBRARY_CONTROLLER_H
