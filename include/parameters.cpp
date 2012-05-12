#include "parameters.h"

namespace bof
{
namespace parameters
{

////////////////////////////////////////////////////////////////////////
//ClusteringParameters
////////////////////////////////////////////////////////////////////////

ClusteringParameters::ClusteringParameters()
{
    setDefault();
}

ClusteringParameters::ClusteringParameters(const ClusteringParameters &cpy)
{
    distance = cpy.distance;
    method = cpy.method;
    numClusters = cpy.numClusters;
    numPass = cpy.numPass;
    branching = cpy.branching;
    FLANNmethod = cpy.FLANNmethod;
    cbIndex = cpy.cbIndex;
}

ClusteringParameters::ClusteringParameters(int _numClusters,
                                         int _numPass,
                                         char _method,
                                         char _distance,
                                         cvflann::flann_centers_init_t _FLANNmethod,
                                         int _branching,
                                         float _cbIndex)
{
    distance = _distance;
    method = _method;
    numClusters = _numClusters;
    numPass = _numPass;
    branching = _branching;
    FLANNmethod = _FLANNmethod;
    cbIndex = _cbIndex;
}

ClusteringParameters& ClusteringParameters::operator=(const ClusteringParameters &rhs)
{
    if(this == &rhs)
        return *this;

    distance = rhs.distance;
    method = rhs.method;
    numClusters = rhs.numClusters;
    numPass = rhs.numPass;
    branching = rhs.branching;
    FLANNmethod = rhs.FLANNmethod;
    cbIndex = rhs.cbIndex;

    return *this;
}

void ClusteringParameters::setDefault()
{
    distance = 'e';
    method = 'a';
    numClusters = 500;
    numPass = 5;
    branching = 32;
    FLANNmethod = cvflann::CENTERS_RANDOM;
    cbIndex = 0.2;
}

void ClusteringParameters::set(int _numClusters,
                                 int _numPass,
                                 char _method,
                                 char _distance,
                                 cvflann::flann_centers_init_t _FLANNmethod,
                                 int _branching,
                                 float _cbIndex)
{
    distance = _distance;
    method = _method;
    numClusters = _numClusters;
    numPass = _numPass;
    branching = _branching;
    FLANNmethod = _FLANNmethod;
    cbIndex = _cbIndex;
}

////////////////////////////////////////////////////////////////////////
//SVMParameters
////////////////////////////////////////////////////////////////////////
SVMParameters::SVMParameters()
{
    setDefault();
}

SVMParameters::SVMParameters(const SVMParameters &cpy)
{
    type = cpy.type;
    kernel = cpy.kernel;
    degree = cpy.degree;
    gamma = cpy.gamma;
    coef0 = cpy.coef0;
    C = cpy.C;
    cache = cpy.cache;
    eps = cpy.eps;
    nu = cpy.nu;
    p = cpy.p;
    termType = cpy.termType;
    iterations = cpy.iterations;
    shrinking = cpy.shrinking;
    probability = cpy.probability;
    weight = cpy.weight;
    kFold = cpy.kFold;
}

SVMParameters& SVMParameters::operator=(const SVMParameters &rhs)
{
    if(this == &rhs)
        return *this;

    type = rhs.type;
    kernel = rhs.kernel;
    degree = rhs.degree;
    gamma = rhs.gamma;
    coef0 = rhs.coef0;
    C = rhs.C;
    cache = rhs.cache;
    eps = rhs.eps;
    nu = rhs.nu;
    p = rhs.p;
    termType = rhs.termType;
    iterations = rhs.iterations;
    shrinking = rhs.shrinking;
    probability = rhs.probability;
    weight = rhs.weight;
    kFold = rhs.kFold;

    return *this;
}

void SVMParameters::setDefault()
{
    type = CvSVM::NU_SVC;
    kernel = CvSVM::RBF;
    degree = 3;
    gamma = 1;
    coef0 = 0.5;
    C = 1;
    cache = 256;
    eps = 0.0001;
    nu = 0.5;
    p = 0.2;
    termType = CV_TERMCRIT_ITER+CV_TERMCRIT_EPS;
    iterations = 1000;
    shrinking = 0;
    probability = 0;
    weight = 0;
    kFold = 10;
}

}//namespace parameters
}//namesapce bof
