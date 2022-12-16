//
// Created by Finn Lukas Busch
// finn.lukas.busch@gmail.com
//

#ifndef RLSIMLIBRARY_DISTANCESENSORS_H
#define RLSIMLIBRARY_DISTANCESENSORS_H

#include<SFML/Graphics.hpp>
#include<Eigen/Dense>
#include "util.h"
#include "Obstacle.h"

/**
 * DistanceSensor (and Collision) class.
 */
class DistanceSensors:public sf::Drawable, public sf::Transformable {
private:
    GlobalParams* config;
    int num_ships = 1;
    int num_rays = 8;
    float d_angle = 0;

    Eigen::Array<float, Eigen::Dynamic, 6>* states; // Ship states!
    Eigen::Array<float, Eigen::Dynamic, 4> rays;
    Eigen::Array<float, Eigen::Dynamic, 4> ship_b_boxs;
    Eigen::Matrix<float, 2, 4> ship_corners;
    Eigen::Array<float, Eigen::Dynamic, 1> ray_angles;
    Eigen::Array<float, Eigen::Dynamic, 1> t_rays;
    Eigen::Array<bool, Eigen::Dynamic, 1> col;
    Eigen::Array<float, Eigen::Dynamic, 1> dists;

    sf::VertexArray viz_rays;
    sf::VertexArray viz_obs;
    std::vector<std::vector<Obstacle*>> obs;
public:
    DistanceSensors(GlobalParams* config, int num_rays, int num_ships, float shipw, float shiph, Eigen::Array<float, Eigen::Dynamic, 6>* states, std::vector<std::vector<Obstacle*>> obs);
    void update(float dt);
    bool update_col(int ship_id);
    void update_single(int i);
    void draw(sf::RenderTarget &target, sf::RenderStates r_states) const;
    const Eigen::Array<bool, Eigen::Dynamic, 1>& get_col();
    const Eigen::Array<float, Eigen::Dynamic, 1>& get_dists();

};


#endif //RLSIMLIBRARY_DISTANCESENSORS_H
