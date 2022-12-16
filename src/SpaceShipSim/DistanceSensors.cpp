//
// Created by finn on 10/11/22.
//

#include "DistanceSensors.h"
#include <iostream>
/**
 * Spawn ray
 * @param num_rays: number of rays to spawn in circular direction
 * @param num_ships: number of ships to spawn rays for
 */
DistanceSensors::DistanceSensors(GlobalParams* config, int num_rays, int num_ships, float shipw, float shiph, Eigen::Array<float, Eigen::Dynamic, 6>* states, std::vector<std::vector<Obstacle*>> obs):
        num_rays(num_rays),
        num_ships(num_ships),
        viz_rays(sf::Lines, 2*num_rays*num_ships),
        viz_obs(sf::Lines, 4){
    this->config = config;
    this->states = states;
    this->obs = obs;
    rays.resize(num_rays*num_ships, rays.cols());
    for(int i = 0; i < num_ships; i++){
        std::vector<Obstacle*> obs_;
        obs.push_back(obs_);
    }
    ship_b_boxs.resize(4*num_ships, ship_b_boxs.cols());
    ray_angles.resize(num_rays, ray_angles.cols());
    t_rays.resize(num_ships*num_rays, t_rays.cols());
    col.resize(num_ships, col.cols());
    col.setConstant(false);
    dists.resize(num_ships*num_rays, dists.cols());
    d_angle = 2*M_PI/(float)num_rays;
    viz_obs[0].position = sf::Vector2f(-5, -5);
    viz_obs[1].position = sf::Vector2f(5, -5);
    viz_obs[2].position = sf::Vector2f(5, -5);
    viz_obs[3].position = sf::Vector2f(5, 5);

    ship_corners(0, 0) = -shipw/2;
    ship_corners(1, 0) = -shiph/2;
    ship_corners(0, 1) = shipw/2;
    ship_corners(1, 1) = -shiph/2;
    ship_corners(0, 2) = shipw/2;
    ship_corners(1, 2) = shiph/2;
    ship_corners(0, 3) = -shipw/2;
    ship_corners(1, 3) = shiph/2;

    for(int j = 0; j < num_ships; j++) {
        for (int i = 0; i < num_rays; i++) {
            viz_rays[2 * num_rays * j + 2 * i].color = sf::Color(79, 220, 43, 30);
//            viz_rays[2 * num_rays * j + 2 * i + 1].color =sf::Color(215, 255, 207, 50) ;
            viz_rays[2 * num_rays * j + 2 * i + 1].color =sf::Color(79, 220, 43, 30) ;
            if(j==0){
                ray_angles(i, 0) = d_angle*(float)i;
            }
        }
    }
    // TODO fix size
}



/**
 * Sorter struct class to sort obstacles by distance.
 */
struct Sorter {
    Sorter(float x0, float y0) { this-> x0 =x0; this->y0 = y0;}
    bool operator () (Obstacle* lhs, Obstacle* rhs) {
        float d1 = powf(x0-lhs->get_x(), 2) + powf(y0-lhs->get_y(),2);
        float d2 = powf(x0-rhs->get_x(), 2) + powf(y0-rhs->get_y(),2);
        if(d1<d2){
            return true;
        }
        return false;}

    float x0;
    float y0;
};

/**
 * Atan approx for efficiency.
 */
inline float atan_scalar_approximation(float x) {
    float a1  =  0.99997726f;
    float a3  = -0.33262347f;
    float a5  =  0.19354346f;
    float a7  = -0.11643287f;
    float a9  =  0.05265332f;
    float a11 = -0.01172120f;

    float x_sq = x*x;
    return
            x * (a1 + x_sq * (a3 + x_sq * (a5 + x_sq * (a7 + x_sq * (a9 + x_sq * a11)))));
}

/**
 * atan2 using approximation.
 */
float atan2_auto_2(float x, float y) {
    float pi = M_PI;
    float pi_2 = M_PI_2;
    bool swap = fabs(x) < fabs(y);
    float atan_input = (swap ? x : y) / (swap ? y : x);

    // Approximate atan
    float res = atan_scalar_approximation(atan_input);

    // If swapped, adjust atan output
    res = swap ? (atan_input >= 0.0f ? pi_2 : -pi_2) - res : res;
    // Adjust quadrants
    if      (x >= 0.0f && y >= 0.0f) {}                     // 1st quadrant
    else if (x <  0.0f && y >= 0.0f) { res =  pi + res; } // 2nd quadrant
    else if (x <  0.0f && y <  0.0f) { res = -pi + res; } // 3rd quadrant
    else if (x >= 0.0f && y <  0.0f) {}                     // 4th quadrant
    return res;
}

/**
 * Generic circle + rectangle intersect.
 */
bool circ_rect_intersect(float dx, float dy, float r, float w, float h){
    if (dx > (w/2 + r)) { return false; }
    if (dy > (h/2 + r)) { return false; }

    if (dx <= (w/2)) { return true; }
    if (dy <= (h/2)) { return true; }

    float d = powf(dx - w/2,2) + powf(dy - h/2,2);

    return (d <= powf(r,2));
}

/**
 * Circle touches outside walls?
 */
bool outer_wall_intersect(float x, float y, float r, float w, float h){
    if( abs(x) > w/2-r){return true;}
    if(abs(y) > h/2-r){return true;}
    return false;
}

/**
 * Updates rays + collisions.
 */
void DistanceSensors::update(float dt) {
    Eigen::Matrix<float, 2, 2> rotmat;
    col.setConstant(false); // set collision to false.
    for(int i = 0; i < num_ships; i++) {
        update_single(i);
    }
    if(config->viz) {
        int k;
        for(int i = 0; i < num_ships; i++) {
            const float x1 = states->operator()(i, 0);
            const float y1 = states->operator()(i, 1);
            for (int j = 0; j < num_rays; j++) {
                // TODO Add Ship Angle!
//                const float x2 = cosf(states->operator()(i, 2) + ray_angles(j, 0));
//                const float y2 = sinf(states->operator()(i, 2) + ray_angles(j, 0));
                const float x2 = cosf(0 + ray_angles(j, 0));
                const float y2 = sinf(0 + ray_angles(j, 0));
                k = i*num_rays + j;
                float t_f = t_rays(k, 0);
                viz_rays[k*2].position = sf::Vector2f(x1, y1);
                viz_rays[k*2 + 1].position = sf::Vector2f(x1 + t_f*x2, y1 + t_f*y2);
            }
        }
    }
}


void DistanceSensors::draw(sf::RenderTarget &target, sf::RenderStates r_states) const {
    target.draw(viz_rays, r_states);
}

/**
 * Collision getter
 * @return
 */
const Eigen::Array<bool, Eigen::Dynamic, 1> &DistanceSensors::get_col() {
    return col;
}

/**
 * Ray distance getter
 * @return
 */
const Eigen::Array<float, Eigen::Dynamic, 1> &DistanceSensors::get_dists() {
    return t_rays;
}

/**
 * Update just collisions for each ship.
 * @param ship_id
 * @return
 */
bool DistanceSensors::update_col(int ship_id) {
    col(ship_id, 0) = false;
    const float x1 = states->operator()(ship_id, 0);
    const float y1 = states->operator()(ship_id, 1);
    std::sort(obs[ship_id].begin(), obs[ship_id].end(), Sorter(x1, y1));
    for (auto& ob_ : obs[ship_id]) {
        if(!col(ship_id, 0)) {
            float w = ob_->get_w();
            float h = ob_->get_h();
            if(ob_->is_outer_wall()){
                col(ship_id, 0) = outer_wall_intersect(x1, y1, 2.1, w, h);
            }
            else {
                float obs_ang = -ob_->get_angle();
                float s_o = sinf(obs_ang);
                float c_o = cosf(obs_ang);
                float dx = abs(c_o * (ob_->get_x() - x1) + s_o * (ob_->get_y() - y1));
                float dy = abs(-s_o * (ob_->get_x() - x1) + c_o * (ob_->get_y() - y1));
                col(ship_id, 0) = circ_rect_intersect(dx, dy, 2.1, w, h);
            }
        }
    }
    return col(ship_id, 0);
}

/**
 * Update rays for one ship.
 * @param i: ship id
 */
void DistanceSensors::update_single(int i) {
    // Set Rays to 1000 and get current ship state
    const float x1 = states->operator()(i, 0);
    const float y1 = states->operator()(i, 1);
    t_rays(Eigen::seq(i*num_rays, (i+1)*num_rays-1), 0).setConstant(1000);
    std::sort(obs[i].begin(), obs[i].end(), Sorter(x1, y1)); // Pre sort obstacles
    for (auto& ob_ : obs[i]) {
        auto m = ob_->get_corners();
        int min_id;
        int max_id;
        // Only check if not already colliding.
        if(!col(i, 0)) {
            float w = ob_->get_w();
            float h = ob_->get_h();
            if(ob_->is_outer_wall()){
                col(i, 0) = outer_wall_intersect(x1, y1, 2.1, w, h);
            }
            else {
                float obs_ang = -ob_->get_angle();
                float s_o = sinf(obs_ang);
                float c_o = cosf(obs_ang);
                float dx = abs(c_o * (ob_->get_x() - x1) + s_o * (ob_->get_y() - y1));
                float dy = abs(-s_o * (ob_->get_x() - x1) + c_o * (ob_->get_y() - y1));
                col(i, 0) = circ_rect_intersect(dx, dy, 2.1, w, h);
            }
        }
        // ## Collision check end
        float min_dist = -1;

        // Build spanning angle from obstacle to ship in order to determine rays of interest.
        // Also find min dist between obstacle and ship
        float min_ang = 0;
        float max_ang = 0;

        for(int pa = 0; pa < 4; pa++){
            float dx =  m->operator()(0, pa) - x1;
            float dy = m->operator()(1, pa) - y1;
            if(!ob_->is_outer_wall()) {
                float ang = atan2_auto_2(dx, dy);
                if (ang < 0 && (ob_->get_x() - x1) < 0) {
                    ang += 2 * M_PI;
                }
                if (min_ang == 0 || ang < min_ang) {
                    min_ang = ang;
                }
                if (max_ang == 0 || ang > max_ang) {
                    max_ang = ang;
                }
            }

            float dist = powf(dx, 2) + powf(dy, 2);
            if(min_dist == -1 || dist < min_dist){
                min_dist = dist;
            }
        }
        min_dist = sqrtf(min_dist);
        // Change if rays should rotate with ship.
//        float ship_angle = states->operator()(i, 2); // TODO Temporary
        float ship_angle = 0; // TODO Temporary
        // if outer wall, take all rays
        if(ob_->is_outer_wall()){
            min_id = 0;
            max_id = num_rays-1;
        }
        else {
            float temp_min;
            float temp_max;
            // Angles of rays are always positive
            temp_min = (min_ang - ship_angle);
            temp_max = (max_ang - ship_angle);
            if (temp_min < 0) {
                temp_min += 2 * M_PI;
            }
            if (temp_max < 0) {
                temp_max += 2 * M_PI;
            }
            temp_min = fmodf(temp_min, 2 * M_PI);
            temp_max = fmodf(temp_max, 2 * M_PI);
            if (temp_min > temp_max) {
                float t = temp_min;
                temp_min = temp_max;
                temp_max = t;
            }
            if (abs(temp_max - temp_min) > M_PI) {
                temp_max = temp_max - 2 * M_PI;
            }
            if (temp_min > temp_max) {
                float t = temp_min;
                temp_min = temp_max;
                temp_max = t;
            }
            min_id = (int) floor(temp_min / (2 * M_PI) * (num_rays - 1));
            max_id = (int) ceil(temp_max / (2 * M_PI) * (num_rays - 1));
        }
        // Do the actual raycasting.
        for (int j = min_id; j < max_id+1; j++) {
            int id = j < 0 ? num_rays + j : j;
            // Only interesting if obstacle is closer than current ray collision
            if (t_rays(i * num_rays + id, 0) > min_dist) {

                float angle = ship_angle + ray_angles(id, 0);
                float x2_ = cosf(angle);
                float y2_ = sinf(angle);
                const float x2 = x1 + x2_;
                const float y2 = y1 + y2_;

                // Ray intersection with the 4 obstacle edges.
                for (int p = 0; p < 4; p++) {
                    const float x3 = m->operator()(0, p);
                    const float y3 = m->operator()(1, p);
                    const float x4 = p == 3 ? m->operator()(0, 0) : m->operator()(0, p + 1);
                    const float y4 = p == 3 ? m->operator()(1, 0) : m->operator()(1, p + 1);
                    float t_den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
                    float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / t_den;
                    if (0 <= t && t < t_rays(i * num_rays + id, 0)) {
                        float u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / t_den;
                        if (0 <= u && u <= 1) {
                            t_rays(i * num_rays + id, 0) = t;
                        }
                    }
                }
            }
        }
    }
}

