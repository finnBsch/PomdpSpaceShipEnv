#ifndef RLSIMLIBRARY_RL_AGENT_H
#define RLSIMLIBRARY_RL_AGENT_H

#include <SFML/Graphics.hpp>
#include "Controller.h"
#include <Eigen/Dense>
template <typename s, typename a> class scenario;
/**
 * Generic RL Agent Class. All RL Agents have (at least)
 * - a dynamics function to step forward
 * - an input actuation array
 * - a state array
 * - a draw function
 * @tparam actuation_array actuation array type (usually std::array<float, n> for n inputs)
 * @tparam state_array state array type (usually std::array<float, n> for n state dimension)
 */
template <typename actuation_array, typename state_array> class rl_agent: public sf::Drawable, public sf::Transformable {
protected:
    state_array state;
    std::string agent_name;
    int agent_id;
    float score = 0;
    actuation_array actuation;
public:
    rl_agent(int agent_id);

    const state_array& get_state() const; // State getter
    const actuation_array& get_actuation() const; // State getter
    virtual void update(Eigen::Array<float, Eigen::Dynamic, 6>& states, Eigen::Array<float, Eigen::Dynamic, 4>& actuation, float dt)  = 0; // TODO Make 4 templated
    std::string get_name();
    Controller<actuation_array, state_array>* get_controller();
    virtual void reset() = 0;
    void set_id(int id);
    // Eval
    state_array& get_prev_state();
    void saveState(state_array state);
    void delta_score(float score);
    // Virtual methods
    virtual std::string get_pos_formatted() = 0;
    virtual void step(float dt) = 0; // Virtual step function, defined in subclasses
    virtual void draw(sf::RenderTarget &target, sf::RenderStates states) const = 0;
    virtual ~rl_agent() = default;
};


#endif //RLSIMLIBRARY_RL_AGENT_H
