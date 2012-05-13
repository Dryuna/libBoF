#include "bagoffeatures.h"

namespace bof
{

BoFBuilder::BoFBuilder()
{
    size = 0;
    featureLength = 0;
    built = false;
}

BoFBuilder::BoFBuilder(int _featureLength)
{
    featureLength = _featureLength;
    size = 0;
    built = false;
}

BoFBuilder::BoFBuilder(const BoFBuilder &cpy)
{
    trainFeatures = cpy.trainFeatures;
    codex = cpy.codex;
    size = cpy.size;
    featureLength = cpy.featureLength;
    built = cpy.built;
}

BoFBuilder::~BoFBuilder()
{
    clearAll();
}

void BoFBuilder::addFeature(const std::vector<double> &feature)
{
    FeatureVector f(feature);
    if(size != 0)
        assert((int)f.getSize() == featureLength);
    else
        featureLength = f.getSize();

    f.normalize();
    trainFeatures.push_back(f);
    size++;
}

void BoFBuilder::addFeature(const FeatureVector &feature)
{
    addFeature(feature.data);
}

void BoFBuilder::addFeatures(const std::vector<FeatureVector> &features)
{
    for(size_t i = 0; i < features.size(); ++i)
        addFeature(features[i].data);
}

void BoFBuilder::clearAll()
{
    clearFeatures();
    clearCodex();
}

void BoFBuilder::clearFeatures()
{
    trainFeatures.clear();
    size = 0;
    featureLength = 0;
}

void BoFBuilder::clearCodex()
{
    codex.clear();
    built = false;
}

void BoFBuilder::buildCodex(const parameters::ClusteringParameters &parameters,
                     int method)
{
    if(size > 0)
    {
        built = true;
        codex.cluster(trainFeatures, parameters, method);
    }

}

bool BoFBuilder::getBoF(const std::vector<std::vector<double> > &input, FeatureHistogram &hist, bool normalized)
{
    assert(built);
    if(built)
    {
        int idx;
        hist.resize(codex.getSize());
        hist.zero();
        for(size_t i = 0; i < input.size(); ++i)
        {
            idx = codex.match(input[i]);
            hist.addAt(idx);
        }
        if(normalized)
            hist.normalize();
        return true;
    }
    else
        return false;
}

bool BoFBuilder::getBoF(const std::vector<FeatureVector> &input, FeatureHistogram &hist, bool normalized)
{
    assert(built);
    if(built)
    {
        int idx;
        hist.resize(codex.getSize());
        hist.zero();
        for(size_t i = 0; i < input.size(); ++i)
        {
            idx = codex.match(input[i]);
            hist.addAt(idx);
        }
        if(normalized)
            hist.normalize();
        return true;
    }
    else
        return false;
}

}









// LEGACY CODE:


//#include <stdlib.h>
//#include <time.h>
//#include "bagoffeatures.h"
//
//
//BagOfFeatures::BagOfFeatures()
//{
//    params.numClasses = 0;
//    params.numFeatures = 0;
//    params.numImages = 0;
//    params.featureLength = 0;
//    testObject = NULL;
//    validObject = NULL;
//    trainObject = NULL;
//    data = NULL;
//    hClusterData = NULL;
//    hTree = NULL;
//    SVMModel = NULL;
//    //SVMModel_CV = NULL;
//    //NBModel_CV = NULL;
//}
//
//BagOfFeatures::BagOfFeatures(BoFParameters p, DataSet* val)
//{
//    int i;
//
//    params = p;
//    hClusterData = NULL;
//    hTree = NULL;
//    SVMModel = NULL;
//
//    if(val != NULL)
//    {
//        testObject = new ObjectSet [params.numClasses];
//        validObject = new ObjectSet [params.numClasses];
//        trainObject = new ObjectSet [params.numClasses];
//        data = new DataSet [params.numClasses];
//
//        int train, valid, test, label;
//
//        for(i = 0; i < params.numClasses; i++)
//        {
//            data[i] = val[i];
//            data[i].getDataInfo(train, valid, test, label);
//            if(test > 0)
//                testObject[i].alloc(test);
//            if(valid > 0)
//                validObject[i].alloc(valid);
//            if(train > 0)
//                trainObject[i].alloc(train);
//        }
//    }
//    else
//    {
//        testObject = NULL;
//        validObject = NULL;
//        trainObject = NULL;
//        data = NULL;
//    }
//
//}
//
//BagOfFeatures::~BagOfFeatures()
//{
//    dealloc();
//}
//
//void BagOfFeatures::alloc(BoFParameters p, DataSet* val)
//{
//    int i;
//    dealloc();
//
//    params = p;
//
//    testObject = new ObjectSet [params.numClasses];
//    validObject = new ObjectSet [params.numClasses];
//    trainObject = new ObjectSet [params.numClasses];
//    data = new DataSet [params.numClasses];
//
//    int train, valid, test, label;
//
//    for(i = 0; i < params.numClasses; i++)
//    {
//        data[i] = val[i];
//        data[i].getDataInfo(train, valid, test, label);
//        if(test > 0)
//            testObject[i].alloc(test);
//        if(valid > 0)
//            validObject[i].alloc(valid);
//        if(train > 0)
//            trainObject[i].alloc(train);
//    }
//}
//
//void BagOfFeatures::dealloc()
//{
//    if(data != NULL)
//    {
//        delete [] testObject;
//        delete [] validObject;
//        delete [] trainObject;
//        delete [] data;
//    }
//    if(SVMModel)
//        svm_destroy_model(SVMModel);
//    if(hClusterData != NULL)
//    {
//        delete [] hClusterData;
//        delete [] hTree;
//    }
//
//}
//
//
//void BagOfFeatures::extractFeatures(ImageFeatures &f, cv::Mat img)
//{
//    cv::Mat processed;
//    (*params.preprocess)(img, processed);
//
//    if(params.featureType == FEATURES_SIFT)
//    {
//        f.extractSIFT_CV(processed,
//                    params.siftParams.detectionThreshold,
//                    params.siftParams.edgeThreshold,
//                    params.verbose);
//    }
//    else if(params.featureType == FEATURES_SURF)
//    {
//        f.extractSURF_CV(processed,
//                    params.surfParams.hessianThreshold,
//                    params.surfParams.nOctives,
//                    params.surfParams.nLayers,
//                    params.surfParams.extended,
//                    params.verbose);
//    }
//}
//
//void BagOfFeatures::clusterFeatures()
//{
//    if(params.clusterType == CLUSTERING_K_MEANS)
//    {
//        if(params.verbose)
//            cout << "Clustering using K-Means with " << params.clustParams.numClusters << " clusters..." << endl;
//        codex.buildKClustering(trainObject,
//                            params.numClasses,
//                            params.numFeatures,
//                            params.featureLength,
//                            params.clustParams.numClusters,
//                            params.clustParams.numPass,
//                            params.clustParams.method,
//                            params.clustParams.distance);
//    }
//    else if(params.clusterType == CLUSTERING_FLANN)
//    {
//        if(params.verbose)
//            cout << "Clustering using FLANN with " << params.clustParams.numClusters << " clusters..." << endl;
//        codex.FLANNClustering(trainObject,
//                            params.numClasses,
//                            params.numFeatures,
//                            params.featureLength,
//                            params.clustParams.numClusters,
//                            params.clustParams.branching,
//                            params.clustParams.numPass,
//                            params.clustParams.FLANNmethod,
//                            params.clustParams.cbIndex
//                            );
//    }
//}
//
//void BagOfFeatures::optimizeDictionary()
//{
//    int i, j, k;
//    Dictionary bestCodex;
//    double avgAccuracy;
//    double bestAvgAccuracy = 0;
//    int label;
//
//    i = 0;
//
//    cout << "\nOptimizing the dictionary for the Bag of Features..." << endl;
//
//    while(i < params.optParams.numSteps)
//    {
//        for(j = 0; j < params.optParams.clusterRepeat; ++j)
//        {
//            clusterFeatures();
//            if(params.verbose)
//                cout << "Building the histograms of features..." << endl;
//
//            for(k = 0; k < params.numClasses; ++k)
//            {
//                label = data[k].getLabel();
//                trainObject[k].buildBoFs(codex, label);
//                validObject[k].buildBoFs(codex, label);
//            }
//            //Train the classifier
//            train();
//
//            avgAccuracy = 0;
//            for(k = 0; k < params.numClasses; ++k)
//            {
//                label = data[k].getLabel();
//                avgAccuracy += testSet(validObject[k], label);
//            }
//            avgAccuracy /= (double)params.numClasses;
//            if(avgAccuracy > bestAvgAccuracy)
//            {
//                bestAvgAccuracy = avgAccuracy;
//                bestCodex = codex;
//                saveModel("Models/Optimization");
//            }
//
//            if(params.verbose)
//            {
//                cout << "Average accuracy: " << avgAccuracy
//                    << " (Best so far: " << bestAvgAccuracy << ")" << endl;
//            }
//            cout << endl;
//        }
//        ++i;
//        params.clustParams.numClusters += params.optParams.clusterStep;
//    }
//
//    if(params.verbose)
//        cout << "Optimization completed!" << endl
//            << "Best validation optimization results: " << bestAvgAccuracy << endl
//            << "Keeping the best dictionary..." << endl << endl;
//
//    codex = bestCodex;
//    params.clustParams.numClusters = codex.size;
//}
//
//void BagOfFeatures::trainSVM_CV()
//{
//    int i, j, k, l = -1;
//    int totalData = 0;
//
//    if(params.verbose)
//        cout << "Training SVM Classifier (OpenCV)..." << endl;
//
//    SVMModel_CV.clear();
//
//    //Get the total number of training data
//    for(i = 0; i < params.numClasses; i++)
//        totalData += data[i].getTrainSize();
//
//    cv::Mat trainData(totalData, codex.size, CV_32FC1);
//    cv::Mat dataLabel(totalData, 1, CV_32FC1);
//
//    // For each class
//    for(i = 0; i < params.numClasses; i++)
//    {
//        // Get the number of images
//        int size = data[i].getTrainSize();
//        for(j = 0; j < size; j++)
//        {
//            l++;
//            float* lPtr = dataLabel.ptr<float>(l);
//            lPtr[0] = (float)data[i].getLabel();
//            //cout << lPtr[0] << " ";
//            float* dPtr = trainData.ptr<float>(l);
//            // Copy the histograms
//            for(k = 0; k < codex.size; k++)
//            {
//                dPtr[k] = trainObject[i].histogramSet[j].histogram[k];
//                //cout << dPtr[k] << " ";
//            }
//            //cout << endl;
//        }
//    }
//
//    CvSVMParams SVMParam_CV;
//    SVMParam_CV.svm_type = params.svmParams.type;
//    SVMParam_CV.kernel_type = params.svmParams.kernel;
//    SVMParam_CV.degree = params.svmParams.degree;
//    SVMParam_CV.gamma = params.svmParams.gamma;
//    SVMParam_CV.coef0 = params.svmParams.coef0;
//    SVMParam_CV.C = params.svmParams.C;
//    SVMParam_CV.nu = params.svmParams.nu;
//    SVMParam_CV.p = params.svmParams.p;
//    SVMParam_CV.class_weights = 0;
//    SVMParam_CV.term_crit = cvTermCriteria(params.svmParams.termType,
//                                           params.svmParams.iterations,
//                                           params.svmParams.eps);
//
//    if(!SVMModel_CV.train_auto(trainData,
//                                dataLabel,
//                                cv::Mat(),
//                                cv::Mat(),
//                                SVMParam_CV,
//                                params.svmParams.kFold))
//        cout << "Training failed..." << endl;
//    else if(params.verbose)
//        cout << "Training successful..." << endl;
//
//}
//
//bool BagOfFeatures::trainSVM()
//{
//    if(SVMModel != NULL)
//    {
//        svm_destroy_model(SVMModel);
//        //svm_destroy_param(&SVMParam);
//    }
//
//    if(params.verbose)
//        cout << "Training SVM Classifier (libSVM)..." << endl;
//
//    int i, j, k, l = -1;
//    int totalData = 0;
//    int size, length = codex.size;
//    int count;
//    //Get the total number of training data
//    for(i = 0; i < params.numClasses; i++)
//        totalData += data[i].getTrainSize();
//
//    // Set up the data
//    struct svm_problem SVMProblem;
//    SVMProblem.l = totalData;
//    SVMProblem.y = new double [totalData];
//    SVMProblem.x = new struct svm_node* [totalData];
//
//    // For each class
//    for(i = 0; i < params.numClasses; i++)
//    {
//        // Get the number of images
//        size = data[i].getTrainSize();
//        for(j = 0; j < size; j++)
//        {
//            l++;
//            count = 0;
//            for(k = 0; k < length; k++)
//            {
//                if(trainObject[i].histogramSet[j].histogram[k] != 0)
//                    count++;
//            }
//            SVMProblem.x[l] = new struct svm_node [count+1];
//            count = 0;
//            for(k = 0; k < length; k++)
//            {
//                //cout << trainObject[i].histogramSet[j].histogram[k] << " ";
//                if(trainObject[i].histogramSet[j].histogram[k] != 0)
//                {
//                    SVMProblem.x[l][count].index = k+1;
//                    SVMProblem.x[l][count].value = trainObject[i].histogramSet[j].histogram[k];
//                    count++;
//                }
//            }
//            //cout << endl;
//            SVMProblem.x[l][count].index = -1;
//            SVMProblem.y[l] = data[i].getLabel();
//        }
//    }
//
//    // Types
//    SVMParam.svm_type = params.svmParams.type;
//    SVMParam.kernel_type = params.svmParams.kernel;
//    // Parameters
//    SVMParam.degree = params.svmParams.degree;
//    SVMParam.gamma = params.svmParams.gamma;
//    SVMParam.coef0 = params.svmParams.coef0;
//    SVMParam.C = params.svmParams.C;
//    // For training only
//    SVMParam.cache_size = params.svmParams.cache;
//    SVMParam.eps = params.svmParams.eps;
//    SVMParam.nu = params.svmParams.nu;
//    SVMParam.shrinking = params.svmParams.shrinking;
//    SVMParam.probability = params.svmParams.probability;
//    // Don't change the weights
//    SVMParam.nr_weight = params.svmParams.weight;
//
//
//    double* target = new double [totalData];
//    svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
//    delete [] target;
//
//    if(svm_check_parameter(&SVMProblem, &SVMParam) != NULL)
//    {
//        //svm_cross_validation(&SVMProblem, &SVMParam, 10, target);
//        SVMModel = svm_train(&SVMProblem, &SVMParam);
//        cout << "Training successful!" << endl;
//        return true;
//    }
//    else
//    {
//        cout << "SVM Parameters are not feasible!" << endl;
//        return false;
//    }
//    //svm_save_model("svmSURF800",SVMModel);
//    //classifierType = LIBSVM_CLASSIFIER;
//}
//
//
//void BagOfFeatures::buildBoF()
//{
//    int i;
//
//    params.numFeatures = 0;
//
//    //First extracting the features
//    for(i = 0; i < params.numClasses; ++i)
//    {
//        processDataSet(data[i], i);
//    }
//
//    if(params.verbose)
//        cout << "Total number of training features: " << params.numFeatures << endl;
//
//    //delete [] data;
//
//    //codex.alloc(params.clustParams.numClusters, params.featureLength);
//    //Next building the dictionary
//    if(params.optParams.numSteps)
//        optimizeDictionary();
//    else
//        clusterFeatures();
//
//    codex.calcCentroid();
//
//    cout << "Building the histograms..." << endl;
//    for(i = 0; i < params.numClasses; ++i)
//    {
//        int label = data[i].getLabel();
//        //cout << "Class " << i << "\n\t";
//        trainObject[i].buildBoFs(codex, label);
//        validObject[i].buildBoFs(codex, label);
//        testObject[i].buildBoFs(codex, label);
//    }
//}
//
//void BagOfFeatures::train()
//{
//    if(CLASSIFIER_SVM_CV)
//        trainSVM_CV();
//    else if(CLASSIFIER_SVM)
//        trainSVM();
//    //    cout << "Training Failed Because LibSVM is a horrible library..." << endl;
//}
//
//void BagOfFeatures::testDataSet()
//{
//    for(int i = 0; i < params.numClasses; ++i)
//    {
//        int label = data[i].getLabel();
//        cout << "Training set: ";
//        testSet(trainObject[i], label);
//        cout << "Validation set: ";
//        testSet(validObject[i], label);
//        cout << "Test set: ";
//        testSet(testObject[i], label);
//    }
//}
//
//double BagOfFeatures::classifyImage(cv::Mat img)
//{
//    cv::Mat processed;
//    (*params.preprocess)(img, processed);
//
//    ImageFeatures feat;
//    HistogramFeatures hist;
//
//    //Extract the features
//    extractFeatures(feat, processed);
//    //Build the histogram
//    hist.buildBoF(feat, codex, -1);
//
//    if(CLASSIFIER_SVM_CV)
//        return hist.predict(SVMModel_CV);
//    else if(CLASSIFIER_SVM)
//        return hist.predict(SVMModel);
//    else
//        return 0;
//}
//
//double BagOfFeatures::testSet(ObjectSet obj, int label)
//{
//    double result;
//    if(CLASSIFIER_SVM)
//    {
//        result = obj.predict(SVMModel, label);
//    }
//    else if(CLASSIFIER_SVM_CV)
//    {
//        result = obj.predict(SVMModel_CV, label);
//    }
//
//    if(params.verbose)
//        cout << "Accuracy for " << label << ": " << result << endl;
//
//    return result;
//}
//
//void BagOfFeatures::processDataSet(DataSet set, int obj)
//{
//    int train, valid, test, label;
//    int i;
//    cv::Mat img;
//    set.getDataInfo(train, valid, test, label);
//    for(i = 0; i < train; ++i)
//    {
//        if(params.verbose)
//            cout << "Loading training image: " << set.getDataList(i) << endl;
//        img = cv::imread(set.getDataList(i), 0);
//        extractFeatures(trainObject[obj].featureSet[i], img);
//        params.numFeatures += trainObject[obj].featureSet[i].size;
//    }
//    for(i = 0; i < valid; ++i)
//    {
//        if(params.verbose)
//            cout << "Loading validation image: " << set.getDataList(i+train) << endl;
//        img = cv::imread(set.getDataList(i+train), 0);
//        extractFeatures(validObject[obj].featureSet[i], img);
//    }
//    for(i = 0; i < test; ++i)
//    {
//        if(params.verbose)
//            cout << "Loading test image: " << set.getDataList(i+train+valid) << endl;
//        img = cv::imread(set.getDataList(i+train+valid), 0);
//        extractFeatures(testObject[obj].featureSet[i], img);
//    }
//}
//
//void BagOfFeatures::saveModel(char* name)
//{
//    codex.save(name);
//    char temp[64];
//    if(params.classifierType == CLASSIFIER_SVM_CV)
//    {
//        strcpy(temp, name);
//        strcat(temp, "_SVM");
//        SVMModel_CV.save(temp);
//    }
//}
//
//bool BagOfFeatures::loadModel(char* name)
//{
//    char temp[64];
//    if(params.classifierType == CLASSIFIER_SVM_CV)
//    {
//        strcpy(temp, name);
//        strcat(temp, "_SVM");
//        SVMModel_CV.load(temp);
//    }
//    if(codex.load(name))
//        return true;
//    else
//        return false;
//}
//
