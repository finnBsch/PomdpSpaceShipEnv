//
// Created by Finn Lukas Busch on 9/15/22.
//

#include "SpaceShip.h"
#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <filesystem>
#include "../resources/mainframe.h"
#include "../resources/booster.h"

namespace fs = std::filesystem;
/**
 * Default constructor given controller
 * @param config global config
 * @param controller supplied controller
 * @param params ship params
 * @param x_init start x
 * @param y_init start y
 */
SpaceShip::SpaceShip(GlobalParams* config, ShipParams params,
                     int agent_id):rl_agent<act_arr, state_arr>::rl_agent(agent_id),
                                   upperarm(sf::TriangleStrip,6),
                                   lowerarm(sf::TriangleStrip,6),
                                   glare_left(11),
                                   glare_right(11){
    this->config = config;
    this->params = params;
    I = params.m/12 *(pow(params.l1,2) + pow(params.l2, 2));
    std::cout << "Current path is " << fs::current_path() << '\n';

    if(!mainframe.loadFromMemory(mainframe_png, mainframe_png_len)){
        std::cout << "Warning: Texture not found!" << std::endl;
    }
    if(!booster.loadFromMemory(booster_png, booster_png_len)){
        std::cout << "Warning: Texture not found!" << std::endl;
    }
    if(config->viz) {
        boost_d_x = params.l1;
        if(params.label_ship) {
            label = new Label(0, 0, 50, params.label);
            label->setOrigin(0, 0);
            label->setPosition(0, params.l2 / 2 * 4);
            label->setRotation(180);
        }
        for(int i = 0; i < 6; i++){
            upperarm[i].color = armcolor;
            lowerarm[i].color = armcolor;
        }
        arm_t = params.l2/10;
        arm_d = params.l2/5;
        left_emitter = new emitter(0.1);
        right_emitter = new emitter(0.1);
        float r = 1.75*params.l2/2;
        centerball = new sf::CircleShape(1.75*params.l2/2);
        centerball->setOrigin(centerball->getRadius(), centerball->getRadius());
        centerball->setPosition(0, 0);
//        float ytex = 2.126*1.75*params.l2;
        float w = 1.5*params.l2;
        float scale_tex = (float)mainframe.getSize().y/(float)mainframe.getSize().x;
        centerframe = new sf::RectangleShape(sf::Vector2f(w, scale_tex*w));
        centerframe->setOrigin(w/2, w/2 + w/20);
//        centerframe->setScale(1, -1);
//        centerframe->setFillColor(sf::Color::Transparent);
//        centerframe->setOutlineColor(sf::Color::Transparent);
        centerframe->setTexture(&mainframe);

//        centerframe->setTextureRect(sf::IntRect(-119/2,253/2, 119, 253));
        colorball = new GlowingBall(64, r);
        colorball->setOrigin(0, - w/20);
        frame = new sf::RectangleShape(sf::Vector2f(params.l1, params.l2));
        frame->setOrigin(params.l1/2, params.l2/2);
        frame->setPosition(0, 0);
        centerbar = new sf::RectangleShape(sf::Vector2f(boost_d_x*2, params.l2/5));
        centerbar->setOrigin(boost_d_x, params.l2/10);
        centerbar->setPosition(0, 0);
        leftbooster = new sf::RectangleShape(sf::Vector2f(params.l2*1.5, params.l2/2));
        leftbooster->setOrigin(params.l2/2*1.5, params.l2/4);
        leftbooster->setPosition(-boost_d_x,0);
        leftbooster->setTexture(&booster);
        rightbooster = new sf::RectangleShape(sf::Vector2f(params.l2*1.5, params.l2/2));
        rightbooster->setOrigin(params.l2/2*1.5, params.l2/4);
        rightbooster->setPosition(boost_d_x,0);
        rightbooster->setTexture(&booster);
        col_circ = new sf::CircleShape(this->r);
        col_circ->setOrigin(this->r, this->r);
        col_circ->setFillColor(sf::Color::Transparent);
        col_circ->setOutlineColor(sf::Color::Red);
        col_circ->setOutlineThickness(0.1);
    }
    agent_name = "SpaceShip";
    update_state_array();
}
double constrainAngle(double x){
    x = x*180/M_PI;
    x = fmod(x + 180,360);
    if (x < 0)
        x += 360;
    return (x - 180)*M_PI/180;
}

/**
 * Steps the ship dynamics
 * @param dt Time passed since last update
 */
void SpaceShip::step(float dt) {
    // Update viz objects
    if(config->viz){
        leftbooster->setRotation(-90 + theta1*180/M_PI);
        rightbooster->setRotation(-90 + theta2*180/M_PI);
        this->setPosition(x,y);
        this->setRotation(phi*180/M_PI);
        if(params.label_ship) {
            label->resetRotation(phi * 180 / M_PI);
//            label->setText("Vel: " + std::to_string(sqrt(pow(dx, 2) + pow(dy, 2))));
//            label->setText("Score: " + std::to_string(score));
        }
        left_emitter->set_state(x-params.l1/2*cos(phi) + sin(theta1+phi)*params.l2, y-sin(phi)*params.l1/2 - cos(theta1+phi)*params.l2, -M_PI/2 + theta1 + phi, F1/2, 0.1/(0.01+F1));
        right_emitter->set_state(x+params.l1/2*cos(phi)+ sin(theta2+phi)*params.l2, y+sin(phi)*params.l1/2 - cos(theta2+phi)*params.l2, -M_PI/2 + theta2 + phi, F2/2, 0.1 /(0.01+F2));
        left_emitter->update(dt);
        right_emitter->update(dt);
    }
    update_state_array();
}




// Viz Functions
/**
 * Draw the Space Ship
 * use as sf::RenderWindow.draw(SpaceShip). Will automatically fill the params.
 * @param target
 * @param states
 */
void SpaceShip::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    // apply the transform
    states.transform *= getTransform();

    // our particles don't use a texture
    states.texture = NULL;

    // draw the vertex array
    target.draw(*centerbar, states);
    target.draw(upperarm, states);
    target.draw(lowerarm, states);
//    target.draw(*centerball, states);
    target.draw(*centerframe, states);
    target.draw(*colorball, states);
//    target.draw(*frame, states);
    target.draw(*left_emitter);
    target.draw(*right_emitter);
    target.draw(glare_left,states);
    target.draw(glare_right,states);
    target.draw(*leftbooster, states);
    target.draw(*rightbooster, states);
    target.draw(*col_circ, states);

    if(params.label_ship) {
        target.draw(*label, states);
    }
}

/**
 * Destructor. Makes sure Viz objects are deleted too
 */
SpaceShip::~SpaceShip() {
    if(left_emitter) {
        delete left_emitter;
    }
    if(right_emitter) {
        delete right_emitter;
    }
    if(rightbooster) {
        delete rightbooster;
    }
    if(leftbooster) {
        delete leftbooster;
    }
    if(frame) {
        delete frame;
    }
}

/**
 * Set SpaceShip Color
 * @param col
 */
void SpaceShip::set_color(sf::Color col) {
    frame->setFillColor(col);
    leftbooster->setFillColor(structurecolor);
    rightbooster->setFillColor(structurecolor);
    centerbar->setFillColor(sf::Color(50, 50, 50));
    centerball->setFillColor(sf::Color(220, 220, 220));
    colorball->set_color(sf::Color(118, 245, 98));
}

// Scalar/Array conversion functions
/**
 * Fill in state array from scalars.
 */
void SpaceShip::update_state_array() {
    state[0] = x;
    state[1] = y;
    state[2] = phi;
    state[3] = dx;
    state[4] = dy;
    state[5] = dphi;
}

/**
 * Fill in state scalars from array.
 */
void SpaceShip::update_state_scalars() {
    x = state[0];
    y = state[1];
    phi = state[2];
    dx = state[3];
    dy = state[4];
    dphi = state[5];
}


/**
 * Fill in actuation array from scalars.
 */
void SpaceShip::update_actuation_array() {
    actuation[0] = F1;
    actuation[1] = F2;
    actuation[2] = theta1;
    actuation[3] = theta2;
}

/**
 * Fill in actuation scalars from array.
 */
void SpaceShip::update_actuation() {
    F1 = actuation[0];
    F2 = actuation[1];
    theta1 = actuation[2];
    theta2 = actuation[3];
}

/**
 * Position info string
 * @return string consisting of x and y pos. formatted for printing
 */
std::string SpaceShip::get_pos_formatted() {
    return std::string("x = " + std::to_string(x) + ", y = " + std::to_string(y));
}

void SpaceShip::reset() {
    this->score = 0.0;
}


void SpaceShip::update(Eigen::Array<float, Eigen::Dynamic, 6> &states, Eigen::Array<float, Eigen::Dynamic, 4>& actuation, float dt) {
    t += dt;
    t = fmod(t,flicker_t);
//    float sc = sinf(2*M_PI*t/flicker_t)*0.4 + 1;
    colorball->setScale(0.6, 0.6);
    x = states(agent_id, 0);
    y = states(agent_id, 1);
    phi = states(agent_id, 2);
    dx = states(agent_id, 3);
    dy = states(agent_id, 4);
    dphi = states(agent_id, 5);
    F1 = actuation(agent_id, 0);
    F2 = actuation(agent_id, 1);
    theta1 = actuation(agent_id, 2);
    theta2 = actuation(agent_id, 3);
    update_state_array();
    update_actuation_array();
    // Update actuation arms
    update_arms();
    leftbooster->setRotation(-90 + theta1*180/M_PI);
    rightbooster->setRotation(-90 + theta2*180/M_PI);
    glare_left.setRotation(theta1*180/M_PI);
    glare_right.setRotation(theta2*180/M_PI);
    glare_left.setPosition(leftbooster->getPosition().x + sinf(theta1)*leftbooster->getSize().x/2, leftbooster->getPosition().y - cosf(theta1)*leftbooster->getSize().x/2);
    glare_left.setScale(F1/30, F1/20);
    glare_right.setPosition(rightbooster->getPosition().x + sinf(theta2)*rightbooster->getSize().x/2, rightbooster->getPosition().y - cosf(theta2)*rightbooster->getSize().x/2);
    glare_right.setScale(F2/30, F2/20);
    this->setPosition(x,y);
    this->setRotation(phi*180/M_PI);
    if(params.label_ship) {
        label->resetRotation(phi * 180 / M_PI);
//            label->setText("Vel: " + std::to_string(sqrt(pow(dx, 2) + pow(dy, 2))));
//        label->setText("Score: " + std::to_string(score));
    }
    left_emitter->set_state(x-boost_d_x*cos(phi) + sin(theta1+phi)*params.l2/2*1.5, y-sin(phi)*boost_d_x - cos(theta1+phi)*params.l2/2*1.5,dx, dy,  -M_PI/2 + theta1 + phi, F1, 0.1/(0.01+3*F1));
    right_emitter->set_state(x+boost_d_x*cos(phi)+ sin(theta2+phi)*params.l2/2*1.5, y+sin(phi)*boost_d_x - cos(theta2+phi)*params.l2/2*1.5, dx, dy, -M_PI/2 + theta2 + phi, F2, 0.1 /(0.01+3*F2));
    left_emitter->update(dt);
    right_emitter->update(dt);
}

void SpaceShip::update_arms() {
    upperarm[0].position = sf::Vector2f(-boost_d_x - sinf(theta1)*(arm_d + arm_t/2), cosf(theta1)*(arm_d + arm_t/2));
    upperarm[1].position = sf::Vector2f(-boost_d_x - sinf(theta1)*(arm_d - arm_t/2), cosf(theta1)*(arm_d - arm_t/2));
    upperarm[2].position = sf::Vector2f(0, centerball->getRadius()-arm_t*1.5+arm_t/2);
    upperarm[3].position = sf::Vector2f(0, centerball->getRadius()-arm_t*1.5-arm_t/2);
    upperarm[4].position = sf::Vector2f(+boost_d_x - sinf(theta2)*(arm_d + arm_t/2), cosf(theta2)*(arm_d + arm_t/2));
    upperarm[5].position = sf::Vector2f(+boost_d_x - sinf(theta2)*(arm_d - arm_t/2), cosf(theta2)*(arm_d - arm_t/2));

    lowerarm[0].position = sf::Vector2f(-boost_d_x + sinf(theta1)*(arm_d*2 + arm_t/2), -cosf(theta1)*(arm_d*2 + arm_t/2));
    lowerarm[1].position = sf::Vector2f(-boost_d_x + sinf(theta1)*(arm_d*2 - arm_t/2), -cosf(theta1)*(arm_d*2 - arm_t/2));
    lowerarm[2].position = sf::Vector2f(0, -(arm_d*2 + arm_t/2));
    lowerarm[3].position = sf::Vector2f(0, -(arm_d*2 - arm_t/2));
    lowerarm[4].position = sf::Vector2f(+boost_d_x + sinf(theta2)*(arm_d*2 + arm_t/2), -cosf(theta2)*(arm_d*2 + arm_t/2));
    lowerarm[5].position = sf::Vector2f(+boost_d_x + sinf(theta2)*(arm_d*2 - arm_t/2), -cosf(theta2)*(arm_d*2 - arm_t/2));
}
