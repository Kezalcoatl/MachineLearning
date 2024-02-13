#pragma once
#include <MLLib/TypeDefinitions.h>
#include <dlib/svm/rr_trainer.h>
#include <dlib/svm/empirical_kernel_map.h>
#include <dlib/svm/linearly_independent_subset_finder.h>

namespace Regressors
{
	template <class LinkFunctionType>
	class GKMDecisionFunction
	{
        template <class LinkFunctionType2>
        friend class GKMTrainer;

        typedef typename LinkFunctionType::LinkFunction LinkFunction;
        typedef typename LinkFunctionType::KernelType::KernelFunctionType KernelType;
        typedef typename KernelType::scalar_type ScalarType;
        typedef typename KernelType::sample_type SampleType;

        LinkFunction Link;
        dlib::decision_function<KernelType> DecisionFunction;
    public:
        GKMDecisionFunction() = default;

        GKMDecisionFunction(const LinkFunction link,
            const dlib::decision_function<KernelType> decisionFunction);

        ScalarType operator()(SampleType const& sample) const;

        friend void serialize(GKMDecisionFunction const& item, std::ostream& out)
        {
            serialize(item.Link, out);
            dlib::serialize(item.DecisionFunction, out);
        }

        friend void deserialize(GKMDecisionFunction& item, std::istream& in)
        {
            deserialize(item.Link, in);
            dlib::deserialize(item.DecisionFunction, in);
        }
	};

    template <class LinkFunctionType>
    class GKMTrainer
    {
        typedef typename LinkFunctionType::LinkFunction LinkFunction;
        typedef typename LinkFunctionType::KernelType::KernelFunctionType KernelType;
        typedef typename KernelType::scalar_type ScalarType;
        typedef typename KernelType::sample_type SampleType;

        static ScalarType const SofteningParameter;

        typename LinkFunctionType::OneShotTrainingParams LinkFunctionOneShotTrainingParams;
        KernelType Kern;
        unsigned long MaxBasisFunctions;
        size_t MaxNumIterations;
        ScalarType ConvergenceTolerance;
        ScalarType Lambda;

    public:

        GKMTrainer();

        void SetLinkFunctionParameters(const typename LinkFunctionType::OneShotTrainingParams& linkFunctionOSTrainingParams);

        typename LinkFunctionType::OneShotTrainingParams const& GetLinkFunctionParameters() const;

        KernelType const GetKernel() const;

        void SetKernel(const KernelType& k);

        unsigned long GetMaxNumIterations() const;

        void SetMaxNumIterations(unsigned long maxNumIterations_);

        ScalarType GetConvergenceTolerance() const;

        void SetConvergenceTolerance(ScalarType convergenceTolerance_);

        unsigned long GetMaxBasisFunctions() const;

        void SetMaxBasisFunctions(unsigned long maxBasisSize_);

        void SetLambda(ScalarType lambda_);

        ScalarType const GetLambda() const;

        template <typename in_sample_vector_type, typename in_scalar_vector_type>
        GKMDecisionFunction<LinkFunctionType> const Train(const in_sample_vector_type& x_,
            const in_scalar_vector_type& y_) const;
    };

    template <class LinkFunctionType>
    typename GKMTrainer<LinkFunctionType>::ScalarType const GKMTrainer<LinkFunctionType>::SofteningParameter = 1.e-6;
}

#include "impl/GKMTrainer.hpp"