//
// Created by Finn Lukas Busch
// finn.lukas.busch@gmail.com
//

#ifndef RLSIMLIBRARY_BOOSTERGLARE_H
#define RLSIMLIBRARY_BOOSTERGLARE_H
#include <SFML/Graphics.hpp>

/**
 * Class for visualizing the booster glare
 */
class boosterglare:public sf::Drawable, public sf::Transformable {
private:
    sf::VertexArray verts;
    sf::VertexArray verts_inner;
    int num_verts;
public:
    boosterglare(int num_verts);
    void draw(sf::RenderTarget &target, sf::RenderStates states) const override;
};


#endif //RLSIMLIBRARY_BOOSTERGLARE_H
