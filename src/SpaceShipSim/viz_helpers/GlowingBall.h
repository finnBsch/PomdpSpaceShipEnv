//
// Created by finn on 10/9/22.
//

#ifndef RLSIMLIBRARY_GLOWINGBALL_H
#define RLSIMLIBRARY_GLOWINGBALL_H
#include <SFML/Graphics.hpp>

class GlowingBall:public sf::Drawable, public sf::Transformable {
private:
    sf::VertexArray verts;
    sf::VertexArray verts_inner;
    int num_verts;
public:
    explicit GlowingBall(int num_verts, float r);
    void set_color(sf::Color col);
    void set_scale(float s1, float s2);
    void draw(sf::RenderTarget &target, sf::RenderStates states) const override;
};


#endif //RLSIMLIBRARY_GLOWINGBALL_H
