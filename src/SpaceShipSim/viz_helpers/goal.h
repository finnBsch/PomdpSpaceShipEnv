//
// Created by finn on 10/8/22.
//

#ifndef RLSIMLIBRARY_GOAL_H
#define RLSIMLIBRARY_GOAL_H
#include <SFML/Graphics.hpp>
#include <deque> // For pts
#include "../util.h"

class goal:public sf::Drawable {
private:
    GlobalParams* config;
    sf::CircleShape circ;
    float radius;
    int num_pts = 50;
    sf::Color col;
    float spawn_delay = 0.01;
    float time;
    float life_time = 0.2;
    float length = 0;
    sf::VertexArray vertices;
    std::deque<std::array<float, 5>> trajectory; // x, y, thickness, angle, length
public:
    explicit goal(GlobalParams* config, float radius);
    void reset(float x, float y);
    void update(float x, float y, float dt);
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;


};


#endif //RLSIMLIBRARY_GOAL_H
