#ifndef BOF_PARAMETERS_H
#define BOF_PARAMETERS_H

#include <cv.h>
#include <cxcore.h>
#include <ml.h>

namespace bof
{
namespace parameters
{
class ClusteringParameters
{
    public:
        ~ClusteringParameters(){};
        ClusteringParameters(const ClusteringParameters &cpy);
        ClusteringParameters(int _numClusters = 500,
                             int _numPass = 5,
                             char _method = 'a',
                             char _distance = 'e',
                             cvflann::flann_centers_init_t _FLANNmethod = cvflann::CENTERS_RANDOM,
                             int _branching = 32,
                             float _cbIndex = 0.2);

        ClusteringParameters& operator=(const ClusteringParameters &rhs);

        void setDefault();
        void set(int _numClusters = 500,
                 int _numPass = 5,
                 char _method = 'a',
                 char _distance = 'e',
                 cvflann::flann_centers_init_t _FLANNmethod = cvflann::CENTERS_RANDOM,
                 int _branching = 32,
                 float _cbIndex = 0.2);

        int numClusters;
        int numPass;
        char method;
        char distance;
        //For FLANN
        cvflann::flann_centers_init_t FLANNmethod;
        int branching;
        float cbIndex;
};

class SIFTParameters
{
    public:
        ~SIFTParameters(){};
        SIFTParameters(const SIFTParameters &cpy);
        SIFTParameters(int _numFeatures = 0,
                        int _numOctaves = 3,
                        double _contrastThreshold = 0.04,
                        double _edgeThreshold = 10.0,
                        double _sigma = 1.6);

        SIFTParameters& operator=(const SIFTParameters &rhs);

        void setDefault();
        void set(int _numFeatures = 0,
                int _numOctaves = 3,
                double _contrastThreshold = 0.04,
                double _edgeThreshold = 10.0,
                double _sigma = 1.6);

        int numFeatures;
        int numOctaves;
        double contrastThreshold;
        double edgeThreshold;
        double sigma;
};

struct SURFParameters
{
    double hessianThreshold;
    int nOctives;
    int nLayers;
    bool extended;
};

class SVMParameters
{
    public:
        ~SVMParameters(){};
        SVMParameters(const SVMParameters &cpy);
        SVMParameters(int _type = CvSVM::NU_SVC,
                int _kernel = CvSVM::RBF,
                double _degree = 3,
                double _gamma = 1,
                double _coef0 = 0.5,
                double _C = 1,
                double _cache = 256,
                double _eps = 0.0001,
                double _nu = 0.5,
                double _p = 0.2,
                int _termType = CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                int _iterations = 1000,
                int _shrinking = 0,
                int _probability = 0,
                int _weight = 0,
                int _kFold = 10);


        SVMParameters& operator=(const SVMParameters &rhs);

        void setDefault();
        void set(int _type = CvSVM::NU_SVC,
                int _kernel = CvSVM::RBF,
                double _degree = 3,
                double _gamma = 1,
                double _coef0 = 0.5,
                double _C = 1,
                double _cache = 256,
                double _eps = 0.0001,
                double _nu = 0.5,
                double _p = 0.2,
                int _termType = CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,
                int _iterations = 1000,
                int _shrinking = 0,
                int _probability = 0,
                int _weight = 0,
                int _kFold = 10);

        int type;
        int kernel;
        double degree;
        double gamma;
        double coef0;
        double C;
        double cache;
        double eps;
        double nu;
        double p;
        int termType;
        int iterations;
        int shrinking;
        int probability;
        int weight;
        int kFold;
};

struct OptimizationParameters
{
    int clusterRepeat;
    int clusterStep;
    int numSteps;
};

class PreprocessBaseFunction
{
    public:
        virtual void operator()(cv::Mat input, cv::Mat &output)=0;
};

class BoFParameters
{
    public:
        //Preprocessing images
        PreprocessBaseFunction *preprocess;

        //Feature Parameters
        SIFTParameters siftParams;
        SURFParameters surfParams;

        //Clustering parameters
        ClusteringParameters clustParams;

        //Classifier parameters
        SVMParameters svmParams;

        //For optimization
        OptimizationParameters optParams;

        int numClasses;
        int numImages;
        int numFeatures;
        int featureLength;

        int classifierType;
        int clusterType;
        int featureType;

        bool verbose;
};


}//namespace params
}//namespace bof

#endif
