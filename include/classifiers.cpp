#include "classifiers.h"

namespace bof
{
namespace ml
{

////////////////////////////////////////////////////////////////////////
//BaseClassifier
////////////////////////////////////////////////////////////////////////

BaseClassifier::BaseClassifier()
{
    size = 0;
    length = 0;
}

BaseClassifier::~BaseClassifier(){}

BaseClassifier::BaseClassifier(const BaseClassifier &cpy)
{
    trainData = cpy.trainData;
    size = cpy.size;
    length = cpy.length;
}

BaseClassifier& BaseClassifier::operator=(const BaseClassifier &rhs)
{
    if(this == &rhs)
        return *this;
    trainData = rhs.trainData;
    size = rhs.size;
    length = rhs.length;
}

void BaseClassifier::add(const FeatureHistogram &trainFeature)
{
    assert(trainFeature.size);
    if(size == 0)
    {
        length = trainFeature.size;
        trainData.push_back(trainFeature);
    }
    else
    {
        assert(length == trainFeature.size);
        trainData.push_back(trainFeature);
    }
    size++;
}

void BaseClassifier::set(const std::vector<FeatureHistogram> &_trainData)
{
    assert(_trainData.size());
    trainData.clear();
    length = _trainData[0].size;
    for(size_t i = 0; i < _trainData.size(); ++i)
    {
        assert(_trainData[i].size == length);
        trainData.push_back(_trainData[i]);
    }
    size = trainData.size();
}

////////////////////////////////////////////////////////////////////////
//SVMClassifier
////////////////////////////////////////////////////////////////////////

SVMClassifier::SVMClassifier()
{
    parameters::SVMParameters defaultParams;
    setParameters(defaultParams, true);
}

SVMClassifier::SVMClassifier(const SVMClassifier &cpy)
{
    params = cpy.params;
    model = cpy.model;
    autoTrain = cpy.autoTrain;
    kFold = cpy.kFold;
}

SVMClassifier::SVMClassifier(const parameters::SVMParameters &_params, bool _autoTrain)
{
    setParameters(_params, _autoTrain);
}

void SVMClassifier::setParameters(const parameters::SVMParameters &_params, bool _autoTrain)
{
    params.svm_type = _params.type;
    params.kernel_type = _params.kernel;
    params.degree = _params.degree;
    params.gamma = _params.gamma;
    params.coef0 = _params.coef0;
    params.C = _params.C;
    params.nu = _params.nu;
    params.p = _params.p;
    params.class_weights = NULL;
    params.term_crit = cvTermCriteria(_params.termType,
                                       _params.iterations,
                                       _params.eps);

    autoTrain = _autoTrain;
    kFold = _params.kFold;
}

bool SVMClassifier::train()
{
    assert(trainData.size());
    cv::Mat trainDataMat(size, length, CV_32FC1);
    cv::Mat responses(size, 1, CV_32FC1);
    for(size_t i = 0; i < size; ++i)
    {
        responses.at<float>(i,0) = trainData[i].label;
        for(size_t j = 0; j < length; ++j)
        {
            trainDataMat.at<float>(i,j) = trainData[i][j];
        }
    }

    if(autoTrain)
    {
        return model.train_auto(trainDataMat, responses, cv::Mat(), cv::Mat(), params, kFold);
    }
    else
    {
        return model.train(trainDataMat, responses, cv::Mat(), cv::Mat(), params);
    }
}

double SVMClassifier::predict(const FeatureHistogram &hist, bool decisionFunc)
{
    assert(hist.size == length);
    cv::Mat sample(1,length,CV_32F);
    for(size_t i = 0; i < hist.size; ++i)
    {
        sample.at<float>(0,i) = hist[i];
    }
    return model.predict(sample, decisionFunc);
}

void SVMClassifier::save(const std::string &fileName)
{
    model.save(fileName.c_str());
}

void SVMClassifier::load(const std::string &fileName)
{
    model.load(fileName.c_str());
}


} //namespace ml
} //namespace bof
