//
// Created by Finn Lukas Busch on 9/15/22.
//

#ifndef RLSIMLIBRARY_SPACESHIPSIM_H
#define RLSIMLIBRARY_SPACESHIPSIM_H

#include "../scenario.h"
#include "SpaceShip.h"
#include "../Controller.h"
#include "util.h"
#include "SpaceControllers.h"
#include <random>
#include "viz_helpers/goal.h"
#include <eigen3/Eigen/Dense>
#include "DistanceSensors.h"

const int NUM_RAYS = 512;
/**
 * Class for Space Ship Sim
 */
class SpaceShipSim: public  scenario<act_arr, state_arr>{
private:
    act_arr max_in{30, 30, M_PI/3, M_PI/3};
    act_arr min_in{0, 0, -M_PI/3, -M_PI/3};

    void reset_goal(int id);
    void reset_shared_goal();
    std::array<float, 3> deltas = {M_PI/2, M_PI/4, M_PI/4*3};
    std::array<std::array<float, 2>, 5> abs;
    int n_ships;
    sf::Color window_color = sf::Color{33, 29, 36};
    void update_lissajous(float dt);
    Eigen::Array<float, Eigen::Dynamic, 3> dd; // derivative
    Eigen::Array<float, Eigen::Dynamic, 6> prev_states;
    Eigen::Array<float, Eigen::Dynamic, 6> init_states;
    Eigen::Array<float, Eigen::Dynamic, 6> states;
    Eigen::Array<float, Eigen::Dynamic, 4> actuations;
    Eigen::Array<float, Eigen::Dynamic, 1> rewards;
    Eigen::Array<bool, Eigen::Dynamic, 1> dones;
    Eigen::Array<float, Eigen::Dynamic, 1> alivetimes;
    Eigen::Array<float, Eigen::Dynamic, 1> in_goal_time;
    Eigen::Array<float, Eigen::Dynamic, 9> goals; // 3 goal coordinates, x0, y1, (and t2, v3, a4, b5, delta6, vx7, vy8) if dynamic
    Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS> relative_state;
    std::vector<std::vector<Obstacle*>> obs;
    DistanceSensors* rays;
    // Viz
    bool draw_obs = true;
    bool draw_rays = false;
    std::vector<goal*> goals_viz;
    bool label_ships = false;
    std::vector<std::string> ship_labels;
    std::uniform_real_distribution<float>* distrx;
    std::uniform_real_distribution<float>* distry;
    std::uniform_real_distribution<float>* distrphi;
    std::uniform_real_distribution<float>* distrt;
    std::uniform_int_distribution<int>* distrdelta;
    std::uniform_int_distribution<int>* distrab;
    std::uniform_real_distribution<float>* distrv;
    float dist_to_goal(float x, float y, float gx, float gy);
    int frame_id = 0;

public:
    SpaceShipSim(GlobalParams* params, int n_ships, std::vector<std::string> ship_labels = std::vector<std::string>{});
    ~SpaceShipSim();
    void draw(float dt) override;
    // Add Ships
    int add_ship();
    // In/Out Communication
    std::vector<float> get_complete_state(int agent_id);
    Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>& get_relative_state_();
    const Eigen::Array<float, Eigen::Dynamic, 1>& get_rewards();
    const Eigen::Array<bool, Eigen::Dynamic, 1>& get_dones();
    void set_controls(Eigen::Array<float, Eigen::Dynamic, 4>& ins);
    const act_arr* get_max_in(){
        return &max_in;
    };
    const act_arr* get_min_in(){
        return &min_in;
    };
    // Dynamics
    void reset();
    void set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi);
    void set_goal(int id, float x, float y);
    void reset_single(int id);
    void reset_ship_to_init(int id);
    bool step(float dt) override;
    bool step() override;
    float eval_fitness(); // Get Fitness
    void update_rewards(float dt); // Get Reward
    void set_view(float width, float height, float x0, float y0);
    void set_viz(bool draw_rays, bool draw_obs);
    void save_frame(std::string save_path);
};
#endif //RLSIMLIBRARY_SPACESHIPSIM_H
