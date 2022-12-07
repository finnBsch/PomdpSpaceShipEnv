//
// Created by finn on 9/16/22.
//

#ifndef RLSIMLIBRARY_UTIL_H
#define RLSIMLIBRARY_UTIL_H

#include "../scenario.h"
/**
 * Scenario specific config
 */
struct SpaceParams: public GlobalParams {
    float gx = 0;
    float gy = 0;
};

#endif //RLSIMLIBRARY_UTIL_H
