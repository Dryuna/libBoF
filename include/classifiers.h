#ifndef BOF_CLASSIFIERS_H
#define BOF_CLASSIFIERS_H

#include "structures.h"
#include <cv.h>
#include <cxcore.h>
#include <ml.h>

namespace bof
{
namespace ml
{

class BaseClassifier
{
    public:
        BaseClassifier();
        ~BaseClassifier();
        BaseClassifier(const BaseClassifier &cpy);
        BaseClassifier& operator=(const BaseClassifier &rhs);

        virtual bool train()=0;
        virtual double predict(const FeatureHistogram &hist, bool decisionFunc)=0;
        virtual void save(const std::string &fileName)=0;
        virtual void load(const std::string &fileName)=0;

        void add(const FeatureHistogram &trainFeature);
        void set(const std::vector<FeatureHistogram> &_trainData);

    protected:
        std::vector<FeatureHistogram> trainData;
        size_t size;
        size_t length;
};

class SVMClassifier : public BaseClassifier
{
    public:
        SVMClassifier();
        SVMClassifier(const SVMClassifier &cpy);
        SVMClassifier(const parameters::SVMParameters &_params, bool _autoTrain=true);
        ~SVMClassifier(){};

        void setAuto(bool _autoTrain){autoTrain=_autoTrain;}
        bool isAuto(){return autoTrain;}

        void setParameters(const parameters::SVMParameters &_params, bool _autoTrain=true);

        bool train();
        double predict(const FeatureHistogram &hist, bool decisionFunc=true);
        void save(const std::string &fileName);
        void load(const std::string &fileName);

    private:
        CvSVM model;
        CvSVMParams params;
        bool autoTrain;
        int kFold;
};

} //namespace ml
} //namespace bof

#endif
