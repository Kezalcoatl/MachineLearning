#pragma once

namespace Regressors
{
    template <class LinkFunctionType>
    GKMDecisionFunction<LinkFunctionType>::GKMDecisionFunction(const LinkFunction link,
        const dlib::decision_function<KernelType> decisionFunction) :
        Link(link),
        DecisionFunction(decisionFunction)
    {
    }

    template <class LinkFunctionType>
    typename GKMDecisionFunction<LinkFunctionType>::ScalarType GKMDecisionFunction<LinkFunctionType>::operator()(SampleType const& sample) const
    {
        return Link(DecisionFunction(sample));
    }

    template <class LinkFunctionType>
    GKMTrainer<LinkFunctionType>::GKMTrainer() :
            MaxNumIterations(100),
            ConvergenceTolerance(1.e-2),
            MaxBasisFunctions(400),
            Lambda(1.e-6)
    {
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetLinkFunctionParameters(const typename LinkFunctionType::OneShotTrainingParams& linkFunctionOSTrainingParams)
    {
        LinkFunctionOneShotTrainingParams = linkFunctionOSTrainingParams;
    }

    template <class LinkFunctionType>
    typename LinkFunctionType::OneShotTrainingParams const& GKMTrainer<LinkFunctionType>::GetLinkFunctionParameters() const
    {
        return LinkFunctionOneShotTrainingParams;
    }

    template <class LinkFunctionType>
    const typename GKMTrainer<LinkFunctionType>::KernelType GKMTrainer<LinkFunctionType>::GetKernel() const
    {
        return Kern;
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetKernel(const KernelType& kern)
    {
        Kern = kern;
    }

    template <class LinkFunctionType>
    unsigned long GKMTrainer<LinkFunctionType>::GetMaxNumIterations() const
    {
        return MaxNumIterations;
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetMaxNumIterations(unsigned long maxNumIterations_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(maxNumIterations_ > 0,
            "\t void GKMTrainer::SetMaxNumIterations()"
            << "\n\t maxNumIterations_ must be greater than 0"
            << "\n\t maxNumIterations_: " << maxNumIterations_
            << "\n\t this:            " << this
        );

        MaxNumIterations = maxNumIterations_;
    }

    template <class LinkFunctionType>
    typename GKMTrainer<LinkFunctionType>::ScalarType GKMTrainer<LinkFunctionType>::GetConvergenceTolerance() const
    {
        return ConvergenceTolerance;
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetConvergenceTolerance(ScalarType convergenceTolerance_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(convergenceTolerance_ > 0,
            "\t void GKMTrainer::SetConvergenceTolerance()"
            << "\n\t convergenceTolerance_ must be greater than 0"
            << "\n\t convergenceTolerance_: " << convergenceTolerance_
            << "\n\t this:            " << this
        );

        ConvergenceTolerance = convergenceTolerance_;
    }

    template <class LinkFunctionType>
    unsigned long GKMTrainer<LinkFunctionType>::GetMaxBasisFunctions() const
    {
        return MaxBasisFunctions;
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetMaxBasisFunctions(unsigned long maxBasisFunctions_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(maxBasisFunctions_ > 0,
            "\t void GKMTrainer::SetMaxBasisFunctions()"
            << "\n\t maxBasisFunctions_ must be greater than 0"
            << "\n\t maxBasisFunctions_: " << maxBasisFunctions_
            << "\n\t this:            " << this
        );

        MaxBasisFunctions = maxBasisFunctions_;
    }

    template <class LinkFunctionType>
    void GKMTrainer<LinkFunctionType>::SetLambda(ScalarType lambda_)
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(lambda_ >= 0,
            "\t void GKMTrainer::SetLambda()"
            << "\n\t lambda must be greater than or equal to 0"
            << "\n\t lambda_: " << lambda_
            << "\n\t this:   " << this
        );

        Lambda = lambda_;
    }

    template <class LinkFunctionType>
    const typename GKMTrainer<LinkFunctionType>::ScalarType GKMTrainer<LinkFunctionType>::GetLambda() const
    {
        return Lambda;
    }

    template <class LinkFunctionType>
    template <typename in_sample_vector_type, typename in_scalar_vector_type>
    const GKMDecisionFunction<LinkFunctionType> GKMTrainer<LinkFunctionType>::Train(
        const in_sample_vector_type& x_,
        const in_scalar_vector_type& y_) const
    {
        auto x = dlib::mat(x_);
        auto y = dlib::mat(y_);
        // make sure requires clause is not broken
        DLIB_ASSERT(dlib::is_learning_problem(x, y),
            "\t GKMTrainer::Train(x,y)"
            << "\n\t invalid inputs were given to this function"
            << "\n\t is_vector(x): " << dlib::is_vector(x)
            << "\n\t is_vector(y): " << dlib::is_vector(y)
            << "\n\t x.size():     " << x.size()
            << "\n\t y.size():     " << y.size()
        );

        size_t numExamples = x.nr();
        // The first thing we do is make sure we have an appropriate ekm ready for use below.

        dlib::linearly_independent_subset_finder<KernelType> lisf(Kern, MaxBasisFunctions);
        dlib::fill_lisf(lisf, x);
        dlib::empirical_kernel_map<KernelType> ekm;
        ekm.load(lisf);
        size_t const numBasis = ekm.basis_size();

        // Now we project all the x samples into kernel space using our EKM 
        dlib::matrix<ScalarType> proj_x(numExamples, numBasis + 1);
        for (size_t row = 0; row < numExamples; ++row)
        {
            // Note that we also append a 1 to the end of the vectors because this is
            // a convenient way of dealing with the bias term later on.
            proj_x(row, 0) = 1.0;
            dlib::set_subm(proj_x, row, 1, 1, numBasis) = dlib::trans(ekm.project(x(row)));
        }
        col_vector<ScalarType> beta(numBasis + 1);
        col_vector<ScalarType> eta(numExamples);
        col_vector<ScalarType> mu(numExamples);
        col_vector<ScalarType> z = y;

        dlib::matrix<ScalarType> weights = dlib::identity_matrix<ScalarType>(numExamples);
        dlib::matrix<ScalarType> const lambdaMat = dlib::identity_matrix<ScalarType>(numBasis + 1) * Lambda;
        dlib::matrix<ScalarType> trans_proj_x = dlib::trans(proj_x);
        auto convergence = std::numeric_limits<ScalarType>::max();
        size_t iteration = 0;
        LinkFunction link;
        while (iteration < MaxNumIterations && convergence > ConvergenceTolerance)
        {
            beta = dlib::inv(trans_proj_x * weights * proj_x + lambdaMat) * trans_proj_x * weights * z;
            eta = proj_x * beta;
            link = LinkFunctionType::Train(LinkFunctionOneShotTrainingParams, eta, y);
            for (size_t e = 0; e < numExamples; ++e)
            {
                mu(e) = link(eta(e)); // TODO generalise this approach to non-binomial link functions
                weights(e, e) = mu(e) * (1.0 - mu(e));
                z(e) = eta(e) + (y(e) - mu(e)) / (weights(e, e) + SofteningParameter);
            }
            col_vector<ScalarType> betaGradiant = (trans_proj_x * weights) * (proj_x * beta - z);
            convergence = dlib::trans(betaGradiant) * betaGradiant;
            ++iteration;
        }

        // convert the linear decision function into a kernelized one.
        GKMDecisionFunction<LinkFunctionType> df;
        df.DecisionFunction = ekm.convert_to_decision_function(dlib::subm(beta, 1, 0, numBasis, 1));
        df.DecisionFunction.b = -beta(0);
        df.Link = link;

        return df;
    }
}