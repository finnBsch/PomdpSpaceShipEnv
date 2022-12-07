//
// Created by Finn Lukas Busch on 9/15/22.
// Abstract class for RL Sim Scenarios
//
#ifndef RLSIMLIBRARY_SCENARIO_H
#define RLSIMLIBRARY_SCENARIO_H
#include "rl_agent.h"
#include <vector>
#include <SFML/Graphics.hpp>
#include <random>
#include <cxxabi.h>
#include <string>
#include "ThreadPool.h"
#include <map>
//template <typename s, typename a> class rl_agent;

/**
 * Global general simulation params
 */
struct GlobalParams{
    bool viz = false;
    int resx = 1920;
    int resy = 1080;
    float sizex = 170;
    float sizey = 100;
    int print_level = 0; // Print Level - 0: Nothing, 1: Warning, 2: Info, 3: Verbose
    // TODO This is scenario specific, move
    bool auto_reset = false;
    bool dynamic_goal = true;
    bool share_envs = false;
    int num_obstacles = 20;
};

/**
 * Keyboard Input Buffer. Saves pressed and released keys to be used by controllers.
 */
struct KeyInput{
    bool pressed = false;
    bool released = false;
    std::vector<int> pressed_codes;
    std::vector<int> released_codes;
};

template <typename actuation_array, typename state_array> class scenario {
private:
    // Timing
    float last_step = 0.0; // When was the last time step?




protected:
    // Timing
    sf::Clock clock;

    int num_agents = 0;
    GlobalParams* config;
    std::map<int, rl_agent<actuation_array, state_array>*> agents; // Vector to store agents acting
    sf::RenderWindow* window; // RenderWindow for Viz
    sf::View* view;
    KeyInput keys;
    // Strings
    std::string sim_name;
    // Randomizer
    std::random_device rd;
    std::default_random_engine* eng;
    // Threading
    ThreadPool* threadpool;
    // Eval
    bool done = false;


public:
    // Viz
    virtual void draw(float dt) = 0;
    virtual bool step(); // Step Simulation Once
    virtual bool step(float dt);
    int add_agent(rl_agent<actuation_array, state_array>* agent);
    virtual void reset() = 0;

    scenario(GlobalParams* config);
    ~scenario();
    // Get Keyboard
    const KeyInput& get_event();
    // String operations
    std::string& get_name();
    void print_init();

};


#endif //RLSIMLIBRARY_SCENARIO_H
