#ifndef BOF_PARAMETERS_H
#define BOF_PARAMETERS_H

namespace bof
{
namespace parameters
{
class ClusteringParameters
{
    public:
        ClusteringParameters();
        ~ClusteringParameters(){};
        ClusteringParameters(const ClusteringParameters &cpy);
        ClusteringParameters& operator=(const ClusteringParameters &rhs);
        void setDefault();

        int numClusters;
        int numPass;
        char method;
        char distance;
        //For FLANN
        cvflann::flann_centers_init_t FLANNmethod;
        int branching;
        float cbIndex;
};

struct SIFTParameters
{
    double detectionThreshold;
    double edgeThreshold;
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
        SVMParameters();
        ~SVMParameters(){};
        SVMParameters(const SVMParameters &cpy);
        SVMParameters& operator=(const SVMParameters &rhs);
        void setDefault();

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
