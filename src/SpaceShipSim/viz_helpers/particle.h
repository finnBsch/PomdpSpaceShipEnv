//
// Created by finn on 9/15/22.
//

#ifndef RLSIMLIBRARY_PARTICLE_H
#define RLSIMLIBRARY_PARTICLE_H


#include <SFML/Graphics.hpp>

class particle:public sf::Drawable {
private:
    float posx;
    float posy;
    float velx;
    float vely;
    float lifetime;
    float time_passed = 0;
    std::array<int, 3> color;
    sf::CircleShape* circ;
public:
    particle(float posx, float posy, float angle, float vel, float lifetime, std::array<int, 3> color);
    bool update(float dt);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    ~particle();

};


#endif //RLSIMLIBRARY_PARTICLE_H
