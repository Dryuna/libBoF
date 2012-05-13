#ifndef BOF_BAG_OF_FEATURES_H
#define BOF_BAG_OF_FEATURES_H

#include <ml.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "parameters.h"
#include "structures.h"

namespace bof
{
class BoFBuilder
{
    public:
        BoFBuilder();
        BoFBuilder(int _featureLength);
        BoFBuilder(const BoFBuilder &cpy);
        ~BoFBuilder();

        size_t getSize(){return size;}
        void clearAll();
        void clearFeatures();
        void clearCodex();

        void addFeature(const std::vector<double> &feature);
        void addFeature(const FeatureVector &feature);
        void addFeatures(const std::vector<FeatureVector> &features);
        void buildCodex(const parameters::ClusteringParameters &parameters=parameters::ClusteringParameters(),
                     int method=CLUSTERING_FLANN);
        bool isBuilt(){return built;}

        bool getBoF(const std::vector<std::vector<double> > &input, FeatureHistogram &hist, bool normalized=true);
        bool getBoF(const std::vector<FeatureVector> &input, FeatureHistogram &hist, bool normalized=true);
    private:
        std::vector<FeatureVector> trainFeatures;
        size_t size;
        int featureLength;
        FeatureCodex codex;
        bool built;

};

class BoFOptimizer
{
    public:

    private:
};

}

#endif








// LEGACY CODE:

//IplImage* preProcessImages(const IplImage* input, int minSize, int maxSize);
//
//class BagOfFeatures
//{
//    public:
//        BagOfFeatures();
//        BagOfFeatures(BoFParameters p, DataSet* val);
//        ~BagOfFeatures();
//
//        // Allocates the Bag of Features
//        void alloc(BoFParameters p, DataSet* val);
//        void dealloc();
//
//        void extractFeatures(ImageFeatures &f, cv::Mat img);
//        void clusterFeatures();
//
//        void buildBoF();
//        void train();
//        void testDataSet();
//
//        double classifyImage(cv::Mat img);
//
//        void saveModel(char* name);
//        bool loadModel(char* name);
//
//    private:
//
//        void processDataSet(DataSet set, int obj);
//        //Data
//        ObjectSet *testObject;
//        ObjectSet *validObject;
//        ObjectSet *trainObject;
//        DataSet *data;
//
//        BoFParameters params;
//
//        //For Hierarchical Clustering
//        double** hClusterData;
//        Node* hTree;
//
//        //Visual Dictionary
//        //CvMat* dictionary;
//        Dictionary codex;
//        void optimizeDictionary();
//
//        //Classifiers
//        struct svm_parameter SVMParam;
//        struct svm_model *SVMModel;
//        bool trainSVM();
//
//        // The OpenCV algorithms, don't work well right now
//        CvSVM SVMModel_CV;
//        void trainSVM_CV();
//
//        double testSet(ObjectSet obj, int label);
//        //CvNormalBayesClassifier NBModel_CV;
//};
