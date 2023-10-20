//
// Created by finn on 9/15/22.
//

#include "SpaceShipSim.h"
#include <iostream>
#include <utility>
/**
 * Draw the sim. Clear window, render all agents, draw goal, display
 */

std::array<float, 2> get_lissajous(int a, int b, float delta, float t, float scale_x, float scale_y){
    float x = sinf(a * t + delta)*scale_x;
    float y = sinf(b * t) * scale_y;
    return std::array<float, 2>{x, y};
}
void SpaceShipSim::update_lissajous(float dt){
    goals.col(2) = goals.col(2) + goals.col(3)*dt;
    // TODO fix dimensions
    goals.col(0) = Eigen::sin(goals.col(4)*goals.col(2) + goals.col(6))*0.9*config->sizex/2;
    goals.col(1) = Eigen::sin(goals.col(5)*goals.col(2))*0.9*config->sizey/2;
    // vx = vt * scalex * a * cos(a*t + delta)
    // vy = vt * scaley * b * cos(b*t)
    goals.col(7) = goals.col(3) * 0.9*config->sizex/2 * goals.col(4) * Eigen::cos(goals.col(4) * goals.col(2) + goals.col(6));
    goals.col(8) = goals.col(3) * 0.9*config->sizey/2 *goals.col(5) * Eigen::cos(goals.col(5) * goals.col(2));
}

void SpaceShipSim::set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi) {
    states(id, 0) = x;
    states(id, 1) = y;
    states(id, 2) = phi;
    states(id, 3) = vx;
    states(id, 4) = vy;
    states(id, 5) = vphi;
}

void SpaceShipSim::draw(float dt) {
    window->clear(window_color); // make it black

    if(draw_rays) {
        window->draw(*rays);
    }

    if(config->share_envs){
        if(draw_obs) {
            for (auto &ob_: obs[0]) {
                window->draw(*ob_);
            }
        }
        window->draw(*goals_viz[0]);
    }
    else{
        for(int i = 0; i < n_ships; i++){
            if(draw_obs) {
                for (auto &ob_: obs[i]) {
                    window->draw(*ob_);
                }
            }
            window->draw(*goals_viz[i]);
        }
    }
    for(auto const&p:agents){
        auto agent = p.second;
        agent->update(states, actuations, dt);
        window->draw(*agent);
    }
    window->display();
}

// TODO: Fill in Sim specific step function
/**
 * Step with given dt
 * @param dt time step
 */
bool SpaceShipSim::step(float dt) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();
    scenario::step(dt);
    if(config->viz){
        if(keys.pressed) {
            for (auto a: keys.pressed_codes) {
                switch (a) {
                    case sf::Keyboard::O:
                        draw_obs = !draw_obs;
                        break;
                    case sf::Keyboard::R:
                        draw_rays = !draw_rays;
                        break;
                    case sf::Keyboard::Space:
                        reset();
                        break;
                }
            }
        }
    }
    ShipParams s_params;
    alivetimes += dt;
    dones.setConstant(false);
    float I = s_params.m/12 *(powf(s_params.l1,2) + powf(s_params.l2, 2));
    // States = [x, y, phi, dx, dy, phi]
    dd.col(0) = (-Eigen::sin(actuations.col(3) + states.col(2))*actuations.col(1) - Eigen::sin(actuations.col(2) + states.col(2))*actuations.col(0))/s_params.m;
    dd.col(1) = (-s_params.m * s_params.g + Eigen::cos(actuations.col(2) + states.col(2))*actuations.col(0) + Eigen::cos(actuations.col(3) + states.col(2))*actuations.col(1))/s_params.m;
    dd.col(2) = (Eigen::sin(states.col(2)) * s_params.l1/2 * Eigen::sin(actuations.col(3) + states.col(2))*actuations.col(1)
            - Eigen::sin(states.col(2)) * s_params.l1/2 * Eigen::sin(actuations.col(2) + states.col(2))*actuations.col(0)
            + Eigen::cos(states.col(2)) * s_params.l1/2 * Eigen::cos(actuations.col(3) + states.col(2))*actuations.col(1)
            - Eigen::cos(states.col(2)) * s_params.l1/2 * Eigen::cos(actuations.col(2) + states.col(2))*actuations.col(0))/I;
    states.col(3) += dd.col(0) * dt;
    states.col(4) += dd.col(1) * dt;
    states.col(5) += dd.col(2) * dt;

    states.col(0) += states.col(3) * dt;
    states.col(1) += states.col(4) * dt;
    states.col(2) += states.col(5) * dt;

    // Wrap Angle, very important for cost function. Maybe find easier way? This could be inefficient
    states.col(2) = states.col(2)*180/M_PI + 180;
    states.col(2) = states.col(2) - (360 * (states.col(2)/360).cast<int>()).cast<float>();
    auto if_le = states.col(2) < 0;
    states.col(2) = if_le.select(states.col(2) + 360, states.col(2));
    states.col(2) = (states.col(2) - 180)*M_PI/180;
    rays->update(dt);
    if(config->dynamic_goal){
        update_lissajous(dt);
        if(config->viz){
            for(int i=0; i < goals_viz.size(); i++){
                goals_viz[i]->update(goals(i, 0), goals(i, 1), dt);
            }
        }
    }
    update_rewards(dt);
    if(dones.isOnes()){
        done = true;
    }
    if(config->auto_reset){
        if(!config->share_envs) {
            for (int i = 0; i < n_ships; i++) {
                if (dones(i, 0)) {
                    reset_single(i);
                }
            }
        }
        else{
            if(dones.isOnes()){
                reset();
            }
        }
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = (t2 - t1);
    auto aa = ms_double.count();
    if(config->print_level>=3) {
        std::cout << "Stepped in " << aa << "ms\n";
    }
    if(config->viz){
        draw(dt);
    }
    return done;
}

// TODO: Fill in Sim specific step function
/**
 * Step without given dt (i.e. in real time)
 */
bool SpaceShipSim::step() {
    return scenario::step();
}

/**
 * Default Add Ship function. x and y will be randomly sampled. ExternalController.
 */
int SpaceShipSim::add_ship() {
    ShipParams params;
    SpaceShip* ship = new SpaceShip(config, params, num_agents);
    ship->set_color(sf::Color(200, 200, 200));
    return add_agent(ship);

}

/**
 * Get complete world state, extend to include goal pos!
 * @param agent agent of interest
 * @return state of the agent and world
 */
std::vector<float> SpaceShipSim::get_complete_state(int agent_id) {
    auto state = agents.at(agent_id)->get_state();
    std::vector<float> complete_state = {state[0], state[1], state[2], state[3], state[4], state[5]};
    return complete_state;
}

/**
 * Get simplified world state. Everything relative to goal.
 * @param agent agent of interest
 * @return Agent state but relative to goal.
 */
Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>& SpaceShipSim::get_relative_state_() {
    relative_state.col(0) = (goals.col(0) - states.col(0))/config->sizex;
    relative_state.col(1) = (goals.col(1) - states.col(1))/config->sizex;
//    relative_state.col(2) = states.col(0)/config->sizex;
//    relative_state.col(3) = states.col(1)/config->sizey;
    relative_state.col(2) = Eigen::sin(states.col(2));
    relative_state.col(3) = Eigen::cos(states.col(2));
    relative_state.col(4) = states.col(3);
    relative_state.col(5) = states.col(4);
    relative_state.col(6) = states.col(5);
    relative_state.col(7) = goals.col(7);
    relative_state.col(8) = goals.col(8);
    const Eigen::Array<float, Eigen::Dynamic, 1>& dists = rays->get_dists();
    for(int i = 0; i < n_ships; i++) {
        relative_state(i, Eigen::seq(9, 9 + NUM_RAYS - 1)) = dists(Eigen::seq(i*NUM_RAYS, (i+1)*NUM_RAYS -1), Eigen::all)/config->sizey;
    }
    return relative_state;
}


/**
 * Constructor
 * @param config SpaceParams including some fundamental information on the setup
 */
SpaceShipSim::SpaceShipSim(GlobalParams* config, int n_ships, std::vector<std::string> ship_labels, RewardFunction* rew):scenario<act_arr, state_arr>(config) {
    sim_name = "SpaceShipSim";
    this->ship_labels = ship_labels;
    if(!ship_labels.empty()){
        label_ships = true;
    }
    this->rew = rew;

    distrx = new std::uniform_real_distribution<float>(-config->sizex*0.9/2, config->sizex*0.9/2);
    distry = new std::uniform_real_distribution<float>(-config->sizey*0.9/2, config->sizey*0.9/2);
    distrphi = new std::uniform_real_distribution<float>(-M_PI/2, M_PI/2);
    Obstacle* glob_walls = new Obstacle(config, config->sizex, config->sizey, true);
    for(int j = 0; j < n_ships; j++) {
        std::vector<Obstacle *> obs_;
        obs_.push_back(glob_walls);
        obs.push_back(obs_);
    }
    if(config->share_envs) {
        for (int i = 0; i < config->num_obstacles; i++) {
            Obstacle *ob = new Obstacle(config, 5, 5);
            ob->set_state( (float)distrx->operator()(*eng), (float)distry->operator()(*eng), (float)distrphi->operator()(*eng));
            for(int j = 0; j < n_ships; j++) {
                obs[j].push_back(ob);
            }
        }
    }
    else{
        for (int i = 0; i < config->num_obstacles; i++) {
            for(int j = 0; j < n_ships; j++) {
                Obstacle *ob = new Obstacle(config, 5, 5);
                ob->set_state( (float)distrx->operator()(*eng), (float)distry->operator()(*eng), (float)distrphi->operator()(*eng));
                obs[j].push_back(ob);
            }
        }
    }
    rays = new DistanceSensors(config, 512, n_ships, 2, 1, &states, obs);
    prev_states.resize(n_ships, prev_states.cols());
    alivetimes.resize(n_ships, alivetimes.cols());
    prev_actuations.resize(n_ships, prev_actuations.cols());
    prev_states.setZero();
    states.resize(n_ships, states.cols());
    states.setZero();
    dd.resize(n_ships, 3);
    relative_state.resize(n_ships, relative_state.cols());
    actuations.resize(n_ships, actuations.cols());
    actuations.setZero();
    rewards.resize(n_ships, rewards.cols());
    rewards.setZero();
    dones.resize(n_ships, dones.cols());
    dones.setConstant(false);
    in_goal_time.resize(n_ships, in_goal_time.cols());
    in_goal_time.setZero();
    goals.resize(n_ships, goals.cols());
    goals.setZero();
    this->n_ships = n_ships;
    for(int i = 0; i < n_ships; i++){
        if(!config->share_envs || i == 0) {
            goals_viz.push_back(new goal(config, 0.5));
        }
        if(config->viz){
            ShipParams params;
            params.label_ship = label_ships;
            if(label_ships) {
                params.label = ship_labels[i];
            }
            SpaceShip* ship = new SpaceShip(config, params, num_agents);
            ship->set_color(sf::Color::White);
            add_agent(ship);
        }

    }
    if(config->dynamic_goal) {
        distrt = new std::uniform_real_distribution<float>(-M_PI/2, M_PI/2);
        distrdelta = new std::uniform_int_distribution<int>(0, 2);
        distrab = new std::uniform_int_distribution<int>(0, 4);
        distrv = new std::uniform_real_distribution<float>(-0.1, 0.1);
        abs[0] = std::array<float, 2>{1, 1};
        abs[1] = std::array<float, 2>{1, 3};
        abs[2] = std::array<float, 2>{3, 4};
        abs[3] = std::array<float, 2>{3, 5};
        abs[4] = std::array<float, 2>{5, 4};
    }
    reset();
    print_init();
}

/**
 * Evaluate the Fitness. Can be always positive, used for genetic algorithms
 * @return fitness
 */
float SpaceShipSim::eval_fitness() {
    return 0;
}

void SpaceShipSim::update_rewards(float dt){
    auto crashed = rays->get_col() || states.col(0)>config->sizex/2 || states.col(0) < -config->sizex/2 ||
            states.col(1) > config->sizey/2 || states.col(1) < -config->sizey/2;
    auto dists = Eigen::sqrt(Eigen::pow(goals.col(0)-states.col(0),2) + Eigen::pow(goals.col(1) - states.col(1), 2));
    auto prev_dists = Eigen::sqrt(Eigen::pow(goals.col(0)-prev_states.col(0),2) + Eigen::pow(goals.col(1) - prev_states.col(1), 2));


    auto in_goal = (dists < 3); // &&  Eigen::abs(states.col(5)) < M_PI/2; // && Eigen::abs(states.col(3) - goals.col(7)) < 1.2 && Eigen::abs(states.col(4) - goals.col(8)) < 1.2 && Eigen::abs(states.col(5)) < 0.15); //&& Eigen::abs(states.col(2)) < M_PI/5);
    in_goal_time = in_goal.select(in_goal_time + dt, 0);
    auto goal_reached = in_goal_time > 1;
    rewards.setZero();
    // Current rewards
    if (rew->abs_angular_v != 0){
        rewards = rewards - Eigen::abs(states.col(5))*dt*rew->abs_angular_v;
    }
    if (rew->dist != 0){
        rewards = rewards - dists*rew->dist*dt;
    }
    if (rew->abs_angle != 0){
        rewards = rewards - Eigen::abs(states.col(2))*dt*rew->abs_angle;
    }
    if (rew->abs_force != 0){
        rewards = rewards
                - (Eigen::abs(actuations.col(0)) + Eigen::abs(actuations.col(1)))*dt*rew->abs_force;
    }

    // Delta Rewards
    if (rew->delta_dist != 0){
        rewards = rewards + (prev_dists-dists)*rew->delta_dist;
    }
    if (rew->delta_force != 0){
        rewards = rewards - (Eigen::square(actuations.col(0) - prev_actuations.col(0)) + Eigen::square(actuations.col(1) - prev_actuations.col(1)))*rew->delta_force;
    }
    if (rew->delta_thrust_angle != 0){
        rewards = rewards - (Eigen::square(actuations.col(2) - prev_actuations.col(2)) + Eigen::square(actuations.col(3) - prev_actuations.col(3)))*rew->delta_thrust_angle;
    }

    dones = crashed || goal_reached || dones || alivetimes > 20;


    rewards = crashed.select(Eigen::Array<float, Eigen::Dynamic, 1>::Constant(n_ships, 1, -rew->crash), rewards);
    rewards = goal_reached.select(Eigen::Array<float, Eigen::Dynamic, 1>::Constant(n_ships, 1, rew->goal_reached), rewards);
    prev_states = states;
    prev_actuations = actuations;
}

void SpaceShipSim::reset_goal(int id) {
    float x;
    float y;
    if(config->dynamic_goal){
        float t = (float)distrt->operator()(*eng);
        int ab_id = distrab->operator()(*eng);
        int delta_id = distrdelta->operator()(*eng);
        auto ab = abs[ab_id];
        auto delta = deltas[delta_id];
        float v = distrv->operator()(*eng);
        goals(id, 2) = t;
        goals(id, 3) = v;
        goals(id, 4) = ab[0];
        goals(id, 5) = ab[1];
        goals(id, 6) = delta;
        auto xy = get_lissajous(ab[0], ab[1], delta, t, 0.9*config->sizex/2, 0.9*config->sizey/2);
        x = xy[0];
        y = xy[1];
        goals(id, 0) = x;
        goals(id, 1) = y;
    }
    else {
        float d = 0;
        // MAke sure goal is well reachable
        while(d < 25){
            d =25;
            x = (float) distrx->operator()(*eng);
            y = (float) distry->operator()(*eng);
            for(auto& ob_:obs[id]){
                if(!ob_->is_outer_wall()) {
                    d = std::min(d, powf(ob_->get_x() - x, 2) + powf(ob_->get_y() - y, 2));
                }
            }
        }
        goals(id, 0) = x;
        goals(id, 1) = y;
    }
    if(config->viz){
        goals_viz[id]->reset(x, y);
    }
}
float SpaceShipSim::dist_to_goal(float x, float y, float gx, float gy) {
    return sqrtf(powf(x-gx, 2) + powf(y - gy, 2));
}

void SpaceShipSim::reset_single(int id) {
    done = false;
    states.row(id).setZero();
    in_goal_time.row(id).setZero();
    actuations.row(id).setZero();
    alivetimes.row(id).setZero();
    for(auto &ob_ : obs[id]){
        ob_->set_state( (float)distrx->operator()(*eng), (float)distry->operator()(*eng), (float)distrphi->operator()(*eng));
    }
    reset_goal(id);
    float x;
    float y;
    float phi = (float)distrphi->operator()(*eng);
    bool not_found = true;
    while(not_found){
        x = (float)distrx->operator()(*eng);
        y = (float)distry->operator()(*eng);
        states(id, 0) = x;
        states(id, 1) = y;
        bool crashed = rays->update_col(id);
        if(!crashed){
            not_found = false;
        }
    }
    states(id, 0) = x;
    states(id, 1) = y;
    states(id, 2) = phi;
    prev_states.row(id) = states.row(id);
    prev_actuations.row(id).setZero();
    rays->update_single(id);
}

void SpaceShipSim::reset_shared_goal(){
    float x;
    float y;
    if(config->dynamic_goal){
        float t = (float)distrt->operator()(*eng);
        int ab_id = distrab->operator()(*eng);
        int delta_id = distrdelta->operator()(*eng);
        auto ab = abs[ab_id];
        auto delta = deltas[delta_id];
        float v = distrv->operator()(*eng);
        goals.col(2) = t;
        goals.col(3) = v;
        goals.col(4) = ab[0];
        goals.col(5) = ab[1];
        goals.col(6) = delta;
        auto xy = get_lissajous(ab[0], ab[1], delta, t, config->sizex/2, config->sizey/2);
        x = xy[0];
        y = xy[1];
        goals.col(0) = x;
        goals.col(1) = y;
    }
    else {
        x = (float) distrx->operator()(*eng);
        y = (float) distry->operator()(*eng);
        goals.col(0) = x;
        goals.col(1) = y;
    }
    if(config->viz){
        goals_viz[0]->reset(x, y);
    }
}

void SpaceShipSim::reset() {
    if(!config->share_envs) {
        for (int i = 0; i < n_ships; i++) {
            reset_goal(i);
        }
    }
    else{
        reset_shared_goal();
    }
    alivetimes.setZero();
    done = false;
    states.setZero();
    in_goal_time.setZero();
    actuations.setZero();
    rewards.setZero();
    dones.setConstant(false);
    for(int i = 0; i < n_ships; i++){
        float x = (float)distrx->operator()(*eng);
        float y = (float)distry->operator()(*eng);
        float phi = (float)distrphi->operator()(*eng);
        bool not_found = true;
        while(dist_to_goal(x, y, goals(i, 0), goals(i, 1)) < 5 || not_found){
            x = (float)distrx->operator()(*eng);
            y = (float)distry->operator()(*eng);
            states(i, 0) = x;
            states(i, 1) = y;
            bool crashed = rays->update_col(i);
            if(!crashed){
                not_found = false;
            }
        }
        states(i, 0) = x;
        states(i, 1) = y;
        states(i, 2) = phi;
    }
    prev_states = states;
    prev_actuations.setZero();
    init_states = states;
    clock.restart();
}

SpaceShipSim::~SpaceShipSim() {
    if(config->viz) {
        for (int i = 0; i < n_ships; i++) {
            delete (goals_viz[i]);
        }
    }
    delete(distrx);
    delete(distry);
    delete(distrphi);
}

void SpaceShipSim::set_controls(Eigen::Array<float, Eigen::Dynamic, 4>& ins) {
    actuations = ins;
}

const Eigen::Array<float, Eigen::Dynamic, 1>& SpaceShipSim::get_rewards() {
    return rewards;
}

const Eigen::Array<bool, Eigen::Dynamic, 1>& SpaceShipSim::get_dones() {
    return dones;
}

void SpaceShipSim::set_view(float width, float height, float x0, float y0) {
    view->setCenter(x0, y0);
    view->setSize(width, -height);
    window->setView(*view);
}

void SpaceShipSim::set_goal(int id, float x, float y) {
    goals(id, 0) = x;
    goals(id, 1) = y;
    goals_viz.at(id)->reset(x, y);
}


void SpaceShipSim::save_frame(std::string save_path) {
    sf::Texture frame;
    frame.create(config->resx, config->resy);
    frame.update(*window);
    auto content = frame.copyToImage();
    content.saveToFile( save_path+ std::to_string(frame_id) + ".jpg");
    frame_id += 1;
}

void SpaceShipSim::set_viz(bool draw_rays, bool draw_obs) {
    this->draw_rays = draw_rays;
    this->draw_obs = draw_obs;
}

void SpaceShipSim::reset_ship_to_init(int id) {
    states.row(id) = init_states.row(id);
}


