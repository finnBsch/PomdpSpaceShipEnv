//
// Created by Finn Lukas Busch on 9/15/22.
// SpaceShip Agent class
//

#ifndef RLSIMLIBRARY_SPACESHIP_H
#define RLSIMLIBRARY_SPACESHIP_H
#include "../rl_agent.h"
#include <SFML/Graphics.hpp>
#include <array>
#include "util.h"
#include "viz_helpers/emitter.h"
#include "../Label.h"
#include "viz_helpers/boosterglare.h"
#include "viz_helpers/GlowingBall.h"


using act_arr = std::array<float, 4>;
using state_arr = std::array<float, 6>;

struct ShipParams{
    // SpaceShip params
    float m = 1;
    float l1 = 2;
    float l2 = 1;
    float g = 9.81;
    bool label_ship = false;
    std::string label = "No Name";
};

class SpaceShip: public rl_agent<act_arr, state_arr>{
private:

    // Dynamics
    GlobalParams* config;
    ShipParams params;
    float I = 0;
    float x = 0; // xpos
    float y = 0; // ypos
    float dx = 0; // xvel
    float dy = 0; // yvel
    float phi = 0; // angle
    float dphi = 0; // angular vel
    float r = 2.1;
    // Controls
    float F1 = 0; // force actuator 1
    float F2 = 0; // force actuator 2
    float theta1 = 0; // angle actuator 1
    float theta2 = 0; // angle actuator 2
    void update_state_array();
    void update_state_scalars();
    void update_actuation_array();
    void update_actuation();

    // Viz
    float t = 0;
    float flicker_t = 2;
    void update_arms();
    float boost_d_x = 0;
    float arm_t = 0;
    float arm_d = 0;
    emitter* left_emitter = nullptr;
    emitter* right_emitter= nullptr;
    sf::RectangleShape* frame= nullptr;
    sf::RectangleShape* centerbar= nullptr;
    sf::RectangleShape* leftbooster= nullptr;
    sf::RectangleShape* rightbooster= nullptr;
    sf::RectangleShape* centerframe = nullptr;
    GlowingBall* colorball;
    sf::VertexArray upperarm;
    sf::VertexArray lowerarm;
    sf::Texture mainframe;
    sf::Texture booster;
    sf::CircleShape* centerball= nullptr;
    sf::CircleShape* col_circ = nullptr;
//    sf::VertexArray colorball;
    boosterglare glare_left;
    boosterglare glare_right;
    sf::Color structurecolor = sf::Color{120, 120, 120};
    sf::Color armcolor = sf::Color{200, 200, 200};
    Label* label;

public:
    void reset();
    void step(float dt) override;
    SpaceShip(GlobalParams* config, ShipParams params,int agent_id);
    ~SpaceShip() override;
    void update(Eigen::Array<float, Eigen::Dynamic, 6>& states, Eigen::Array<float, Eigen::Dynamic, 4>& actuation, float dt) override;
    // Viz
    void draw(sf::RenderTarget &target, sf::RenderStates states) const override;
    void set_color(sf::Color col);
    // Strings
    std::string get_pos_formatted() override;
};


#endif //RLSIMLIBRARY_SPACESHIP_H
