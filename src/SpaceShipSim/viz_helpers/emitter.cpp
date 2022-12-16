//
// Created by finn on 9/15/22.
//

#include "emitter.h"
#include <cmath>
/**
 * Particle Emitter constructor
 * @param lifetime particle lifetime
 */
emitter::emitter(float lifetime):
        rng(rd()),
        uni(0, colors.size()),
        distangle(0, 0.15),
        disttime(1, 0.5)
{
    this->lifetime = lifetime;
}

/**
 * Set Emitter State
 * @param posx position x
 * @param posy position y
 * @param angle angle relative to ship
 * @param vel particle velocity
 * @param spawn_delay spawn delay (smaller delay = more particles)
 */
void emitter::set_state(float posx, float posy, float angle, float vel, float spawn_delay) {
    this->posx = posx;
    this->posy = posy;
    emit_angle = angle;
    emit_vel = vel;
    this->spawn_delay = spawn_delay;
}

void emitter::set_state(float posx, float posy, float velx_ship, float vely_ship, float angle, float vel, float spawn_delay) {
    this->posx = posx;
    this->posy = posy;
    float velx = velx_ship + cosf(angle)*vel;
    float vely = vely_ship + sinf(angle)*vel;
    emit_angle = atan2f(vely, velx);
    emit_vel = sqrtf(powf(velx, 2) + powf(vely, 2));
    this->spawn_delay = spawn_delay;
}

/**
 * Step forward. Emit particles according to dynamics
 * @param dt time passed
 */
void emitter::update(float dt) {
    dt_temp += dt;
    for(auto it = std::begin(particles); it != std::end(particles); ){
        if((*it)->update(dt)){
            delete *it;
            particles.erase(it);
        }
        else{
            ++it;
        }
    }
    if(dt_temp >= spawn_delay){
        auto color = colors[uni(rng)];
        float sampled_ang = distangle(rng);
        particle* part = new particle(posx, posy, emit_angle+sampled_ang, emit_vel, disttime(rng)*lifetime/(1+ fabsf(sampled_ang)), color);
        particles.push_back(part);
        dt_temp = 0;
    }
}

/**
 * Destructor. Make sure all particles are gone
 */
emitter::~emitter() {
    for(particle* part:particles){
        delete part;
    }
}

/**
 * Draw particles
 * @param target
 * @param states
 */
void emitter::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    for(particle* part:particles){
        target.draw(*part);
    }
}
