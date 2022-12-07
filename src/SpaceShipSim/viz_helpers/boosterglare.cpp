//
// Created by finn on 10/9/22.
//

#include "boosterglare.h"
#include <cmath>

/**
 * constructor
 * @param num_verts: number of verts to draw the shape
 */
boosterglare::boosterglare(int num_verts):
    verts(sf::TriangleFan, num_verts + 1),
    verts_inner(sf::TriangleFan, num_verts + 1){
    this->num_verts = num_verts;
    verts[0].position = sf::Vector2f(0,0);
    verts[0].color = sf::Color(255, 244, 89);
    verts_inner[0].position = sf::Vector2f(0,0);
    verts_inner[0].color = sf::Color(255, 255, 255);
    float r = 0.7;
    float d_angle = -M_PI/(num_verts-1);
    for(int i = 0; i < num_verts; i++){
        verts[i+1].position = sf::Vector2f(cosf(d_angle*i)*r, sinf(d_angle*i)*r*3);
        verts[i+1].color = sf::Color::Transparent;
        verts_inner[i+1].position = sf::Vector2f(cosf(d_angle*i)*r*0.4, sinf(d_angle*i)*r*3*0.4);
        verts_inner[i+1].color = sf::Color(255, 244, 89, 110);
    }
}


/**
 * Draw the shape with correct transform.
 * @param target
 * @param states
 */
void boosterglare::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    states.transform *= getTransform();

    // our particles don't use a texture
    states.texture = NULL;

    // draw the vertex array
    target.draw(verts, states);
    target.draw(verts_inner, states);
}
