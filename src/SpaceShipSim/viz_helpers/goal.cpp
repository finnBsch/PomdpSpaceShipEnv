//
// Created by finn on 10/8/22.
//

#include "goal.h"
#include <cmath>
#include <iostream>


/**
 * Goal constructor. Passes global config and desired radius.
 */
goal::goal(GlobalParams* config, float radius):
    circ(radius=radius),
    radius(radius),
    vertices(sf::TriangleStrip, num_pts*2),
    time(spawn_delay)
{
    this->config = config;
    circ.setOrigin(radius, radius);
}

/**
 * Update the goal point
 * @param x
 * @param y
 * @param dt
 */
void goal::update(float x, float y, float dt) {
    circ.setPosition(x, y);
    time += dt;
    if(config->dynamic_goal) {
        if (time > spawn_delay) {
            time -= spawn_delay;
            if (trajectory.size() == num_pts) {
                length -= trajectory[trajectory.size() - 2][4];
                trajectory.pop_back();
            }
            if (trajectory.empty()) {
                trajectory.push_front(std::array<float, 5>{x, y, radius, 0, 0});
            } else {
                float ang = atanf((y - trajectory[0][1]) / (x - trajectory[0][0]));
                auto len = sqrtf(powf(y - trajectory[0][1], 2) + powf(x - trajectory[0][0], 2));
                length += len;
                trajectory.push_front(std::array<float, 5>{x, y, radius, ang, len});
                if (trajectory.size() >= 3) {
                    trajectory[1][3] = atanf((y - trajectory[2][1]) / (x - trajectory[2][0]));
                }
            }
            if (trajectory.size() * 2 != vertices.getVertexCount()) {
                vertices.resize(trajectory.size() * 2);
            }
            float running_len = 0;
            float fac = radius / length;
            float fac2 = 255 / length;
            for (int i = 0; i < trajectory.size(); i++) {
                if (i > 0) {
                    running_len += trajectory[i - 1][4];
                    trajectory[i][2] = (radius - fac * running_len);
                    vertices[i * 2].color = sf::Color(255, 255, 255, 255 - fac2 * running_len);
                    vertices[i * 2 + 1].color = sf::Color(255, 255, 255, 255 - fac2 * running_len);
                }
                vertices[i * 2].position = sf::Vector2f(trajectory[i][0] - sinf(trajectory[i][3]) * trajectory[i][2],
                                                        trajectory[i][1] + cosf(trajectory[i][3]) * trajectory[i][2]);
                vertices[i * 2 + 1].position = sf::Vector2f(
                        trajectory[i][0] + sinf(trajectory[i][3]) * trajectory[i][2],
                        trajectory[i][1] - cosf(trajectory[i][3]) * trajectory[i][2]);
            }
        }
    }
}

/**
 * Do the drawing, draw the point and trace.
 * @param target
 * @param states
 */
void goal::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    target.draw(circ, states);
    if(config->dynamic_goal) {
        if (trajectory.size() >= 2) {
            target.draw(vertices, states);
        }
    }
}

void goal::reset(float x, float y) {
    time = spawn_delay;
    length = 0;
    circ.setPosition(x, y);
    trajectory.clear();
}

