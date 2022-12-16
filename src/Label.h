//
// Created by finn on 9/20/22.
//

#ifndef RLSIMLIBRARY_LABEL_H
#define RLSIMLIBRARY_LABEL_H
#include <SFML/Graphics.hpp>

/**
 * Generic Label class
 */
class Label: public sf::Drawable, public sf::Transformable{
private:
    float posx;
    float posy;
    float ang;
    int fontsize;
    sf::Text* text;
public:
    Label();
    Label(float posx, float posy);
    Label(float posx, float posy, int fontsize);
    Label(float posx, float posy, int fontsize, const std::string& content);
    void draw(sf::RenderTarget &target, sf::RenderStates states) const override;
    void setText(std::string content);
    void resetRotation(float angle);
    void resetCenter();

};


#endif //RLSIMLIBRARY_LABEL_H
