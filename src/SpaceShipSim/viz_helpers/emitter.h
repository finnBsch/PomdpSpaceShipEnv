//
// Created by finn on 9/15/22.
//

#ifndef RLSIMLIBRARY_EMITTER_H
#define RLSIMLIBRARY_EMITTER_H


#include <queue>
#include <SFML/Graphics.hpp>
#include "particle.h"
#include <random>


class emitter:public sf::Drawable {
private:
    float spawn_delay = 0;
    float lifetime = 0;
    float posx = 0;
    float posy = 0;
    float emit_angle = 0;
    float emit_vel = 0;
    float dt_temp = 0;
    std::vector<std::array<int, 3>> colors = {
            std::array<int, 3>{255, 227, 84},
            std::array<int, 3>{207, 33, 21},
            std::array<int, 3>{247, 163, 17},
            std::array<int, 3>{255, 170, 0}
    };

    std::random_device rd;     // Only used once to initialise (seed) engine
    std::mt19937 rng;    // Random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni; // Guaranteed unbiased
    std::normal_distribution<float> distangle; // Guaranteed unbiased
    std::normal_distribution<float> disttime; // Guaranteed unbiased
    std::vector<particle*> particles;
public:
    emitter(float lifetime);
    void set_state(float posx, float posy, float angle, float vel, float spawn_delay);
    void set_state(float posx, float posy, float velx_ship, float vely_ship, float angle, float vel, float spawn_delay);
    void update(float dt);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    ~emitter();
};



#endif //RLSIMLIBRARY_EMITTER_H
