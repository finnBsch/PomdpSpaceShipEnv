//
// Created by finn on 9/15/22.
//

#include "particle.h"
#include <cmath>

/**
 * Particle constructor
 * @param posx init pos x
 * @param posy init pos y
 * @param angle velocity angle
 * @param vel velocity
 * @param lifetime lifetime
 */
particle::particle(float posx, float posy, float angle, float vel, float lifetime, std::array<int, 3> color) {
    this->posx = posx;
    this->posy = posy;
    this->color = color;
    velx = cosf(angle)*vel;
    vely = sinf(angle)*vel;
    this->lifetime = lifetime;
    circ = new sf::CircleShape(0.1);
    circ->setOrigin(0.1, 0.1);
    circ->setPosition(posx, posy);
}

/**
 * Step forward, move particles and vanish
 * @param dt time passed
 * @return bool if particle should die
 */
bool particle::update(float dt) {
    posx += velx*dt;
    posy += vely*dt;
    circ->setPosition(posx, posy);
    time_passed += dt;
    auto min_ = std::min(std::min(color[0], color[1]), color[2]);
    circ->setFillColor(sf::Color(color[0] - (color[0]-min_)*time_passed/lifetime, color[1] - (color[1]-min_)*time_passed/lifetime, color[2] - (color[2]-min_)*time_passed/lifetime, 255-255*time_passed/lifetime));
    return time_passed >= lifetime;
}

/**
 * destructor, kill sfml circle shape
 */
particle::~particle() {
    delete circ;
}

/**
 * Draw Particle
 * @param target standard sfml input
 * @param states standard sfml input
 */
void particle::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    target.draw(*circ, states);
}
