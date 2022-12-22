//
// Created by finn on 9/19/22.
//

#ifndef RLSIMLIBRARY_INTERFACE_H
#define RLSIMLIBRARY_INTERFACE_H

#include "SpaceShipSim/SpaceShipSim.h"
#include "SpaceShipSim/SpaceControllers.h"
#include "scenario.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <iostream>
#include <iomanip>

namespace py = pybind11;


class SpaceShipInterface{
private:
    SpaceShipSim* sim;
    GlobalParams config;
    RewardFunction rew;
    Eigen::Array<float, Eigen::Dynamic, 4> control_in;
    int n_ships = 1;
    std::vector<std::string> labels;
public:
    SpaceShipInterface(GlobalParams config, int n_ships, RewardFunction rew = RewardFunction(), py::list ns = py::list());
    ~SpaceShipInterface();
    bool step();
    bool step_dt(float dt);
    void reset();
    const act_arr* get_max_in();
    const act_arr* get_min_in();
    Eigen::Array<float, Eigen::Dynamic, 9 + NUM_RAYS>& get_states();
    const Eigen::Array<float, Eigen::Dynamic, 1>& get_rewards();
    const Eigen::Array<bool, Eigen::Dynamic, 1>& get_dones();
    void set_controls(py::array &control_ins);
    void set_view(float width, float height, float x0, float y0);
    void draw(float dt);
    void set_ship(int id, float x, float y, float phi, float vx, float vy, float vphi);
    void set_goal(int id, float x, float y);
    void export_frame(std::string save_path);
    void set_viz(bool draw_rays, bool draw_obs);
    void reset_to_init(int id);
};

template<typename T> std::string print_element(T t, const int& width){
    std::stringstream ret_string;
    ret_string << std::left << std::setw(width) << std::setfill(' ') << t;
    return ret_string.str();
}
int name_width = 40;
int content_width = 15;
std::string float_with_prec(float in, int prec){
    std::stringstream ret_string;
    ret_string << std::fixed << std::setprecision(prec) << in;
    return ret_string.str();
}
PYBIND11_MODULE(pomdp_spaceship_env, m)
{
    py::options options;
    options.disable_function_signatures();
    py::class_<RewardFunction>(m, "RewardFunction",  R"pbdoc(
    Reward Function datastruct.

    Attributes:
	    DistancePenalty (float) : Penalty weight for distance to goal point. Defaults to ``0.0``
	    AbsAnglePenalty (float) : Penalty weight for absolute deviation from horizontal orientation. Defaults to ``0.0``
	    AbsAngleVPenalty (float) : Penalty weight for absolute angular velocity. Defaults to ``10.0``
	    AbsThrustPenalty (float) : Penalty weight for absolute thrust value. Defaults to ``0.0``
	    DeltaThrustAngle (float) : Penalty weight for squared difference between previous and current thruster angle. Defaults to ``0.0``
	    DeltaThrust (float) : Penalty weight for squared difference between previous and current thrust. Defaults to ``0.0``
	    DeltaDistanceReward (float) : Reward weight for difference previous and current distance to goal point. Defaults to ``5.0``
	    CrashPenalty (float) : Terminal penalty for crashing. Defaults to ``1000.0``
	    GoalReward (float) : Terminal reward for reaching the goal. Defaults to ``1000.0``
    )pbdoc")
    .def(py::init(),  R"pbdoc(
    Default Constructor with the parameters listed above.
    )pbdoc")
    .def("__repr__",
          [](const RewardFunction &a) {
              return ("<pomdp_spaceship_env.RewardFunction>: \n" +
              print_element(" ", name_width) + print_element("Dist", content_width) + print_element("Angle", content_width)
              + print_element("Thrust", content_width) + print_element("ThrustAngle", content_width) + "\n" +
                      print_element(std::string(name_width + content_width*4, '-'), name_width+ content_width*4) + "\n" +
              print_element("Weight for current Value", name_width) + print_element(a.dist, content_width) + print_element(a.abs_angle, content_width)
              + print_element(a.abs_force, content_width) + print_element("-", content_width) + "\n" +

              print_element("DeltaWeight/VelocityWeight", name_width) + print_element(a.delta_dist, content_width) + print_element("-", content_width) +
                      print_element(a.delta_force, content_width) + print_element(a.delta_thrust_angle, content_width) + "\n\n"
                      + "Terminal Rewards: Goal: " + float_with_prec(a.goal_reached, 2) + " Crash: " +  float_with_prec(-a.crash, 2)
              );
          }
            )
    .def_readwrite("DistancePenalty", &RewardFunction::dist)
    .def_readwrite("AbsAnglePenalty", &RewardFunction::abs_angle)
    .def_readwrite("AbsAngleVPenalty", &RewardFunction::abs_angular_v)
    .def_readwrite("AbsThrustPenalty", &RewardFunction::abs_force)
    .def_readwrite("DeltaThrustAngle", &RewardFunction::delta_thrust_angle)
    .def_readwrite("DeltaThrust", &RewardFunction::delta_force)
    .def_readwrite("DeltaDistanceReward", &RewardFunction::delta_dist)
    .def_readwrite("CrashPenalty", &RewardFunction::crash)
    .def_readwrite("GoalReward", &RewardFunction::goal_reached);

    py::class_<GlobalParams>(m, "Config", R"pbdoc(
    Confic datastruct.
    
    Attributes:
	    ResX (int) : Width Resolution. Defaults to ``1920``
	    ResY (int) : Height Resolution. Defaults to ``1080``
   	    SizeX (float) : Environment Width in meters. Defaults to ``170``
	    SizeY (float) : Height Resolution. Defaults to ``1080``
	    PrintLevel(int) : -/-.
	    AutoReset (bool) : A boolean determining whether a ship should be reset on terminal state. Defaults to ``True``
	    DynamicGoals (bool) : -/-. Defaults to ``False``
	    ShareEnvs (bool) : -/-. Defaults to ``False``
	    Viz (bool) : -/-. Defaults to ``True``
	    NumObs (int) : -/-. Defaults to ``0``
    )pbdoc")
    .def(py::init(),  R"pbdoc(
    Default Constructor with the parameters listed above.
    )pbdoc")
    .def_readwrite("Viz", &GlobalParams::viz)
    .def_readwrite("ResX", &GlobalParams::resx)
    .def_readwrite("ResY", &GlobalParams::resy)
    .def_readwrite("SizeX", &GlobalParams::sizex)
    .def_readwrite("SizeY", &GlobalParams::sizey)
    .def_readwrite("PrintLevel", &GlobalParams::print_level)
    .def_readwrite("AutoReset", &GlobalParams::auto_reset)
    .def_readwrite("DynamicGoals", &GlobalParams::dynamic_goal)
    .def_readwrite("ShareEnvs", &GlobalParams::share_envs)
    .def_readwrite("NumObs", &GlobalParams::num_obstacles);

    py::class_<SpaceShipInterface>(m, "Env", R"pbdoc(
        Environment class. Contains all the required methods to run and interact with the
        Environment.
        )pbdoc")
        .def(py::init<GlobalParams, int, RewardFunction, py::list>(), R"pbdoc(
        __init__(self: pomdp_spaceship_env.Env, conf: pomdp_spaceship_env.Config, n_ships: int, RewardFunction: pomdp_spaceship_env.RewardFunction = RewardFunction(), labels: list[string] = [])

        Constructor for the Environment with Ship Labels

        Args:
            config (Config) : Environment Parameters
            n_ships (int) : Number of Ships. Defaults to ``1``
            labels (list of Strings) : Labels for the Ships, used for viz.
        )pbdoc", py::arg("conf"), py::arg("n_ships"), py::arg("RewardFunction") = RewardFunction(), py::arg("labels") = py::list())

        .def("Reset", &SpaceShipInterface::reset, R"pbdoc(
        Reset(self: pomdp_spaceship_env.Env)

        Resets the entire environment. All goal points are regenerated, all ships reinitialized and all obstacles regenerated.
        )pbdoc")

        .def("Step", (&SpaceShipInterface::step), R"pbdoc(
        Step(self: pomdp_spaceship_env.Env)

        Steps the environment. If viz was set to True, it will also draw the Environment.
        )pbdoc")

        .def("Step_dt", (&SpaceShipInterface::step_dt), R"pbdoc(
        Step(self: pomdp_spaceship_env.Env, dt: float)

        Steps the environment given an elapsed time. If viz was set to True, it will also draw the Environment.

        Args:
            dt (float) : elapsed time.
        )pbdoc", py::arg("dt"))

        .def("GetState", &SpaceShipInterface::get_states, R"pbdoc(
        GetState(self: pomdp_spaceship_env.Env) -> numpy.ndarray[numpy.float32[n_ships, 521]]

        Get an array of the current states of all ships.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, state_dim)
        )pbdoc")

        .def("SetControl", &SpaceShipInterface::set_controls, R"pbdoc(
        SetControl(self: pomdp_spaceship_env.Env, ControlIn: numpy.ndarray)

        Set the control inputs for all shapes. ``Must be of type np.float32!``

        Args:
        	ControlIn (ndarray) : Numpy Array of Shape (n_ships, control_dim) and type ``np.float32``. Providing an array of a different type yields undesired behavior.
        )pbdoc", py::arg("ControlIn"))

        .def("GetReward", &SpaceShipInterface::get_rewards, R"pbdoc(
        GetReward(self: pomdp_spaceship_env.Env) -> numpy.ndarray[numpy.float32[n_ships, 1]]

        Get an array of the current rewards of all ships.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, 1)
        )pbdoc")

        .def("GetAgentDone", &SpaceShipInterface::get_dones, R"pbdoc(
        GetAgentDone(self: pomdp_spaceship_env.Env) -> numpy.ndarray[numpy.bool[n_ships, 1]]

        Get an boolean array which contains "True" if the corresponding ship has collided or met the goal condition.

        Returns:
            ndarray : Numpy Array of Shape (n_ships, 1)
        )pbdoc")

        .def("GetMinIn", &SpaceShipInterface::get_min_in, R"pbdoc(
        GetMinIn(self: pomdp_spaceship_env.Env) -> List[float[4]]

        Get lower actuation limit of the Environment.

        Returns:
            ndarray : Numpy Array of Shape (control_dim, 1)
        )pbdoc")

        .def("GetMaxIn", &SpaceShipInterface::get_max_in, R"pbdoc(
        GetMaxIn(self: pomdp_spaceship_env.Env) -> List[float[4]]

        Get upper actuation limit of the Environment.

        Returns:
            ndarray : Numpy Array of Shape (control_dim, 1)
        )pbdoc")

        .def("SetView", &SpaceShipInterface::set_view, R"pbdoc(
        SetView(self: pomdp_spaceship_env.Env, width: float, height: float, x0: float, y0: float)

        Set the camera view (viz only).

        Args:
            width (float) : Camera view width.
            height (float) : Camera view height.
            x0 (float) : Camera centerpoint.
            y0 (float) : Camera centerpoint.
        )pbdoc", py::arg("width"), py::arg("height"), py::arg("x0"), py::arg("y0"))

        .def("Draw", &SpaceShipInterface::draw, R"pbdoc(
        Draw(self: pomdp_spaceship_env.Env)

        Draws the Environment without Stepping the Simulation.
        )pbdoc")

        .def("SetShip", &SpaceShipInterface::set_ship, R"pbdoc(
        SetShip(self: pomdp_spaceship_env.Env, id: int, x: float, y: float, phi: float, vx: float, vy: float, vphi: float)

        Set a specific ship, specified by its ID, to a specific Position.

        Args:
            id (int) : Ship Identifier
            x (float) : x coordinate
            y (float) : y coordinate
            phi (float) : angle
            vx (float) : x velocity
            vy (float) : y velocity
            vphi (float) : angular velocity
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"), py::arg("phi"), py::arg("vx"), py::arg("vy"), py::arg("vphi"))

        .def("SetGoal", &SpaceShipInterface::set_goal, R"pbdoc(
        SetGoal(self: pomdp_spaceship_env.Env, id: int, x: float, y: float)

        Set a goal point, specified by the ID, to a specific position. Not working for dynamic goal points.

        Args:
            id (int) : Ship Identifier
            x (float) : x coordinate
            y (float) : y coordinate
        )pbdoc", py::arg("id"), py::arg("x"), py::arg("y"))
        .def("ExportFrame", &SpaceShipInterface::export_frame, R"pbdoc(
        ExportFrame(self: pomdp_spaceship_env.Env, path: str)

        Export the current frame to a given path.

        Args:
            path (str) : Export path.
        )pbdoc", py::arg("path"))

        .def("SetViz", &SpaceShipInterface::set_viz, R"pbdoc(
        SetViz(self: pomdp_spaceship_env.Env, draw_rays: bool, draw_obs: bool)

        Disable or enable obstacle drawing and distance sensor ray drawing.
        
        Args:
        	draw_rays (bool) : Draw Rays?
        	draw_obs (bool) : Draw Obs?
        )pbdoc", py::arg("draw_rays"), py::arg("draw_obs"))

        .def("ResetToInit", &SpaceShipInterface::reset_to_init, R"pbdoc(
        ResetToInit(self: pomdp_spaceship_env.Env, id: int)

        Reset a specific ship, specified by its ID, to the initial position.

        Args:
            id (int) : Ship Identifier
        )pbdoc", py::arg("id"));
    m.attr("__version__") = "dev";
}

#endif //RLSIMLIBRARY_INTERFACE_H
