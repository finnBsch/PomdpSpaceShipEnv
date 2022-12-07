//
// Created by finn on 10/9/22.
//

#include "GlowingBall.h"
#include <cmath>


/**
 * Pulsing ball constructor
 */
GlowingBall::GlowingBall(int num_verts, float r):
    verts_inner(sf::TriangleFan,num_verts+1),
    verts(sf::TriangleFan,num_verts+1){
        this->num_verts = num_verts;
        verts[0].position = sf::Vector2f(0,0);
        verts_inner[0].position = sf::Vector2f(0,0);
        verts_inner[0].color = sf::Color(255, 255, 255);
        float d_angle = 2*M_PI/(num_verts-1);
        for(int i = 0; i < num_verts; i++){
            verts[i+1].position = sf::Vector2f(cosf(d_angle*i)*r, sinf(d_angle*i)*r);
            verts[i+1].color = sf::Color::Transparent;
            verts_inner[i+1].position = sf::Vector2f(cosf(d_angle*i)*r*0.25, sinf(d_angle*i)*r*0.25);
            verts_inner[i+1].color = sf::Color::Transparent;
        }
}

void GlowingBall::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    states.transform *= getTransform();

    // our particles don't use a texture
    states.texture = NULL;
    // draw the vertex array
    target.draw(verts, states);
    target.draw(verts_inner, states);
}

void GlowingBall::set_color(sf::Color col) {
    verts[0].color = col;
    for(int i = 0; i < num_verts; i++){
        verts_inner[i+1].color = col;
    }
}
