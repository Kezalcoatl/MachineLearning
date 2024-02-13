#include "gtest/gtest.h"
#include <MLLib/Regressor.h>
#include <dlib/md5.h>

template <typename T>
std::string GetMD5(T const& item)
{
	using namespace dlib;
	std::stringstream ss;
	serialize(item, ss);
	return dlib::md5(ss);
}

TEST(WrapperSerializationDeserialization, RegressorTests)
{
	/*
	using namespace Regressors;
	typedef col_vector<double> SampleType;
	typedef typename SampleType::type T;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::LinearKernel<SampleType>> LinearKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::PolynomialKernel<SampleType>> PolynomialKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::RadialBasisKernel<SampleType>> RadialBasisKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::SigmoidKernel<SampleType>> SigmoidKRR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::LinearKernel<SampleType>> LinearSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::PolynomialKernel<SampleType>> PolynomialSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::RadialBasisKernel<SampleType>> RadialBasisSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::SigmoidKernel<SampleType>> SigmoidSVR;
	typedef RegressionTypes::RandomForestRegression<KernelTypes::DenseExtractor<SampleType>> DenseRF;

	static size_t const numExamples = 50;
	static size_t const numOrdinates = 10;

	std::vector<SampleType> inputExamples(numExamples, SampleType(numOrdinates));
	std::vector<T> targetExamples(numExamples);
	for (size_t e = 0; e < numExamples; ++e)
	{
		for (size_t o = 0; o < numOrdinates; ++o)
		{
			T sinArg = static_cast<T>(e + 1);
			inputExamples[e](o) = std::sin(sinArg * sinArg) * std::exp(static_cast<T>(o));
		}
		targetExamples[e] = static_cast<T>(e + 1) + std::sin(static_cast<T>(e) * 0.1 * dlib::pi * 2.0);
	}

	std::string const randomSeed = "MLLib";
	ECrossValidationMetric const metric = ECrossValidationMetric::SumSquareMean;
	size_t const numFolds = 4;
	size_t const maxNumCalls = 1000;
	size_t const numThreads = 16;
	std::vector<T> diagnostics;
	std::vector<Regressor<SampleType>> regressors;

	ModifierTypes::NormaliserModifier<SampleType>::OneShotTrainingParams normaliserOSParams;
	ModifierTypes::InputPCAModifier<SampleType>::OneShotTrainingParams PCAOSParams;
	PCAOSParams.TargetVariance = 0.9;
	ModifierTypes::FeatureSelectionModifier<SampleType>::OneShotTrainingParams featureSelectionOSParams;
	featureSelectionOSParams.FeatureFraction = 0.7;

	LinearKRR::OneShotTrainingParams linearKRROSParams;
	linearKRROSParams.MaxBasisFunctions = 400;
	linearKRROSParams.Lambda = 1.e-6;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, linearKRROSParams, normaliserOSParams));

	PolynomialKRR::OneShotTrainingParams polynomialKRROSParams;
	polynomialKRROSParams.MaxBasisFunctions = 400;
	polynomialKRROSParams.Lambda = 1.e-6;
	polynomialKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, polynomialKRROSParams, normaliserOSParams));

	RadialBasisKRR::OneShotTrainingParams radialBasisKRROSParams;
	radialBasisKRROSParams.MaxBasisFunctions = 400;
	radialBasisKRROSParams.Lambda = 1e-6;
	radialBasisKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, radialBasisKRROSParams, normaliserOSParams));

	SigmoidKRR::OneShotTrainingParams sigmoidKRROSParams;
	sigmoidKRROSParams.MaxBasisFunctions = 400;
	sigmoidKRROSParams.Lambda = 1.e-6;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, sigmoidKRROSParams, normaliserOSParams));

	LinearSVR::OneShotTrainingParams linearSVROSParams;
	linearSVROSParams.C = 1.0;
	linearSVROSParams.Epsilon = 1.e-3;
	linearSVROSParams.EpsilonInsensitivity = 0.1;
	linearSVROSParams.CacheSize = 200;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, linearSVROSParams, normaliserOSParams));

	PolynomialSVR::OneShotTrainingParams polynomialSVROSParams;
	polynomialSVROSParams.C = 1.0;
	polynomialSVROSParams.Epsilon = 1.e-3;
	polynomialSVROSParams.EpsilonInsensitivity = 0.1;
	polynomialSVROSParams.CacheSize = 200;
	polynomialSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, polynomialSVROSParams, normaliserOSParams));

	RadialBasisSVR::OneShotTrainingParams radialBasisSVROSParams;
	radialBasisSVROSParams.C = 1.0;
	radialBasisSVROSParams.Epsilon = 1.e-3;
	radialBasisSVROSParams.EpsilonInsensitivity = 0.1;
	radialBasisSVROSParams.CacheSize = 200;
	radialBasisSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, radialBasisSVROSParams, normaliserOSParams));

	SigmoidSVR::OneShotTrainingParams sigmoidSVROSParams;
	sigmoidSVROSParams.C = 1.0;
	sigmoidSVROSParams.Epsilon = 1.e-3;
	sigmoidSVROSParams.EpsilonInsensitivity = 0.1;
	sigmoidSVROSParams.CacheSize = 200;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, sigmoidSVROSParams, normaliserOSParams));

	DenseRF::OneShotTrainingParams denseRFOSParams;
	denseRFOSParams.NumTrees = 1000;
	denseRFOSParams.MinSamplesPerLeaf = 5;
	denseRFOSParams.SubsamplingFraction = 1.0 / 3.0;
	regressors.emplace_back(Regressors::RegressorTrainer::TrainRegressorOneShot<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, diagnostics, denseRFOSParams, normaliserOSParams));

	std::stringstream ss;
	dlib::serialize(regressors, ss);
	std::vector<Regressor<SampleType>> regressors2;
	dlib::deserialize(regressors2, ss);
	EXPECT_EQ(GetMD5(regressors), GetMD5(regressors2));
	*/
}