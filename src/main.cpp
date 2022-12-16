#include <iostream>
#include "SpaceShipSim/SpaceShipSim.h"
#include "SpaceShipSim/SpaceControllers.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace numpy = boost::python::numpy;
using namespace boost::python;
int main() {
    SpaceParams params{true, 1615, 950, 170, 100, 3, true, false, false, 50};
    SpaceShipSim* testSim = new SpaceShipSim(&params, 1);
//    Eigen::Array<float, Eigen::Dynamic, 4> ins;
//    ins.resize(1, 4);
//    ins.setZero();
    for(int i = 0; i < 100000; i++) {
        testSim->step();
        testSim->set_view(170*(1 + sin(i*0.001)), 100, 0, 0);
//        testSim->set_controls(ins);
    }
//
    return 0;
}
