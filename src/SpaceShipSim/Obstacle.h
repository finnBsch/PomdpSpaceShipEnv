//
// Created by finn on 10/11/22.
//

#ifndef RLSIMLIBRARY_OBSTACLE_H
#define RLSIMLIBRARY_OBSTACLE_H

#include <SFML/Graphics.hpp>
#include <Eigen/Dense>
#include "util.h"

/**
 * Rectangular obstacle with 0,0 being the center point
 */
class Obstacle: public sf::Drawable, public sf::Transformable {
private:
    GlobalParams* config;
    float angle = 0;
    float x = 0;
    float y = 0;
    float r = 0;
    float h;
    float w;
    bool global_wall = false;
    Eigen::Matrix<float, 2 ,4> base_corners;
    Eigen::Array<float, 2, 4> corners;
    Eigen::Matrix<float, 2, 2> rotation_matrix;
    sf::RectangleShape rect;
public:
    Obstacle(GlobalParams* config, float sizex, float sizey, bool global_wall = false);
    void set_state(float x, float y, float angle);
    Eigen::Array<float, 2, 4>* get_corners();
    void update();
    const float& get_x();
    const float& get_y();
    const float& get_angle();
    const float& get_w();
    const float& get_h();
    bool is_outer_wall();

    void draw(sf::RenderTarget &target, sf::RenderStates r_states) const override;
};


#endif //RLSIMLIBRARY_OBSTACLE_H
