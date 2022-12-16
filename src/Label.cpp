//
// Created by finn on 9/20/22.
//

#include "Label.h"
#include <iostream>
#include "resources/Arial.h"
Label::Label() {
}

Label::Label(float posx, float posy) {
    this->posx = posx;
    this->posy = posy;
    this->text = new sf::Text();
    text->setFillColor(sf::Color::White);
    sf::Font* font = new sf::Font;
    if(!font->loadFromMemory(Arial_ttf, Arial_ttf_len)){
        std::cout << "Couldn't load font!" << std::endl;
    }
    text->setFont(*font);
//    text->setStyle(sf::Text::Bold);
    text->setScale(-1.0/50, 1.0/50);


}

Label::Label(float posx, float posy, int fontsize):Label::Label(posx, posy) {
    this->fontsize = fontsize;
    text->setCharacterSize(fontsize);
}

Label::Label(float posx, float posy, int fontsize, const std::string& content):Label::Label(posx, posy, fontsize) {
    text->setString(content);
    resetCenter();
}

void Label::draw(sf::RenderTarget &target, sf::RenderStates states) const {
    states.transform *= getTransform();
    states.transform.rotate(ang);
    states.texture = NULL;
    target.draw(*text, states);
}

void Label::setText(std::string content) {
    this->text->setString(content);
    resetCenter();
}

void Label::resetRotation(float angle) {
    ang = -angle;
}

void Label::resetCenter() {
    sf::FloatRect textRect = text->getLocalBounds();
    text->setOrigin(textRect.left + textRect.width/2.0f,
                    textRect.top + textRect.height);
    text->setPosition(0, 0);
}
