//
// Created by finn on 10/11/22.
//

#include "Obstacle.h"

Obstacle::Obstacle(GlobalParams* config, float sizex, float sizey, bool global_wall):
        rect(sf::Vector2f(sizex, sizey)),
        config(config),
        global_wall(global_wall){
    float hw = sizex/2;
    float hl = sizey/2;
    w = sizex;
    h = sizey;
    rect.setOrigin(hw, hl);
    rect.setFillColor(sf::Color(165, 181, 169));
    rotation_matrix.setZero();
    rotation_matrix(0,0) = 1;
    rotation_matrix(1,1) = 1;
    // Lower left corner
    base_corners(0, 0) = - hw;
    base_corners(1, 0) = - hl;
    // Upper left
    base_corners(0, 1) = - hw;
    base_corners(1, 1) = + hl;
    // Upper right
    base_corners(0, 2) = + hw;
    base_corners(1, 2) = + hl;
    // Lower right
    base_corners(0, 3) = + hw;
    base_corners(1, 3) = - hl;
    corners = base_corners;
}

void Obstacle::set_state(float x, float y, float angle) {
    if(!global_wall) {
        this->x = x;
        this->y = y;
        if (r==0) {
            this->r = sqrtf(powf(x, 2) + powf(y, 2));
        }
        if (config->viz) {
            this->setPosition(x, y);
            this->setRotation(angle * 180 / M_PI);
        }
        if (angle != this->angle) {
            rotation_matrix(0, 0) = cosf(angle);
            rotation_matrix(0, 1) = -sinf(angle);
            rotation_matrix(1, 1) = cosf(angle);
            rotation_matrix(1, 0) = sinf(angle);
        }
        this->angle = angle;
        corners = (rotation_matrix * (base_corners));
        corners.row(0) += this->x;
        corners.row(1) += this->y;
    }
}

Eigen::Array<float, 2, 4>* Obstacle::get_corners() {
    return &corners;
}

void Obstacle::draw(sf::RenderTarget &target, sf::RenderStates r_states) const {
    if(!global_wall) {
        r_states.transform *= getTransform();

        r_states.texture = NULL;
        target.draw(rect, r_states);
    }
}

void Obstacle::update() {

}

const float& Obstacle::get_x() {
    return x;
}

const float &Obstacle::get_y() {
    return y;
}

const float &Obstacle::get_angle() {
    return angle;
}

const float &Obstacle::get_w() {
    return w;
}

const float &Obstacle::get_h() {
    return h;
}

bool Obstacle::is_outer_wall() {
    return global_wall;
}
