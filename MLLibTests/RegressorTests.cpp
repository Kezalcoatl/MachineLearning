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

TEST(SerializationDeserialization, RegressorTests)
{
	using namespace Regressors;
	typedef double T;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::LinearKernel<T>> LinearKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::PolynomialKernel<T>> PolynomialKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::RadialBasisKernel<T>> RadialBasisKRR;
	typedef RegressionTypes::KernelRidgeRegression<KernelTypes::SigmoidKernel<T>> SigmoidKRR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::LinearKernel<T>> LinearSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::PolynomialKernel<T>> PolynomialSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::RadialBasisKernel<T>> RadialBasisSVR;
	typedef RegressionTypes::SupportVectorRegression<KernelTypes::SigmoidKernel<T>> SigmoidSVR;
	typedef RegressionTypes::RandomForestRegression<KernelTypes::DenseExtractor<T>> DenseRF;

	static size_t const numExamples = 50;
	static size_t const numOrdinates = 10;

	std::vector<col_vector<T>> inputExamples(numExamples, col_vector<T>(numOrdinates));
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
	CrossValidationMetric const metric = CrossValidationMetric::SumSquareMean;
	size_t const numFolds = 4;
	size_t const maxNumCalls = 1000;
	size_t const numThreads = 16;
	std::stringstream regressorSS;
	Regressor<T> regressor2;

	ModifierTypes::NormaliserModifier<T>::OneShotTrainingParams normaliserOSParams;
	ModifierTypes::InputPCAModifier<T>::OneShotTrainingParams PCAOSParams;
	PCAOSParams.TargetVariance = 0.9;
	ModifierTypes::FeatureSelectionModifier<T>::OneShotTrainingParams featureSelectionOSParams;
	featureSelectionOSParams.FeatureFraction = 0.7;

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::OneShotTrainingParams linearKRROSParams;
	linearKRROSParams.MaxBasisFunctions = 400;
	linearKRROSParams.Lambda = 1.e-6;
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearKRRDiagnostics, linearKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressorSS.clear();
	serialize(linearKRRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(linearKRRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::OneShotTrainingParams polynomialKRROSParams;
	polynomialKRROSParams.MaxBasisFunctions = 400;
	polynomialKRROSParams.Lambda = 1.e-6;
	polynomialKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialKRRDiagnostics, polynomialKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(polynomialKRRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::OneShotTrainingParams radialBasisKRROSParams;
	radialBasisKRROSParams.MaxBasisFunctions = 400;
	radialBasisKRROSParams.Lambda = 1e-6;
	radialBasisKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisKRRDiagnostics, radialBasisKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(radialBasisKRRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::OneShotTrainingParams sigmoidKRROSParams;
	sigmoidKRROSParams.MaxBasisFunctions = 400;
	sigmoidKRROSParams.Lambda = 1.e-6;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidKRRDiagnostics, sigmoidKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(sigmoidKRRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> linearSVRDiagnostics;
	LinearSVR::OneShotTrainingParams linearSVROSParams;
	linearSVROSParams.C = 1.0;
	linearSVROSParams.Epsilon = 1.e-3;
	linearSVROSParams.EpsilonInsensitivity = 0.1;
	linearSVROSParams.CacheSize = 200;
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearSVRDiagnostics, linearSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(linearSVRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(linearSVRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> polynomialSVRDiagnostics;
	PolynomialSVR::OneShotTrainingParams polynomialSVROSParams;
	polynomialSVROSParams.C = 1.0;
	polynomialSVROSParams.Epsilon = 1.e-3;
	polynomialSVROSParams.EpsilonInsensitivity = 0.1;
	polynomialSVROSParams.CacheSize = 200;
	polynomialSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialSVRDiagnostics, polynomialSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(polynomialSVRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::OneShotTrainingParams radialBasisSVROSParams;
	radialBasisSVROSParams.C = 1.0;
	radialBasisSVROSParams.Epsilon = 1.e-3;
	radialBasisSVROSParams.EpsilonInsensitivity = 0.1;
	radialBasisSVROSParams.CacheSize = 200;
	radialBasisSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisSVRDiagnostics, radialBasisSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(radialBasisSVRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(radialBasisSVRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::OneShotTrainingParams sigmoidSVROSParams;
	sigmoidSVROSParams.C = 1.0;
	sigmoidSVROSParams.Epsilon = 1.e-3;
	sigmoidSVROSParams.EpsilonInsensitivity = 0.1;
	sigmoidSVROSParams.CacheSize = 200;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidSVRDiagnostics, sigmoidSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(sigmoidSVRRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), GetMD5(regressor2));

	regressorSS.clear();
	std::vector<T> denseRFDiagnostics;
	DenseRF::OneShotTrainingParams denseRFOSParams;
	denseRFOSParams.NumTrees = 1000;
	denseRFOSParams.MinSamplesPerLeaf = 5;
	denseRFOSParams.SubsamplingFraction = 1.0 / 3.0;
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, denseRFDiagnostics, denseRFOSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	serialize(denseRFRegressor, regressorSS);
	deserialize(regressor2, regressorSS);
	EXPECT_EQ(GetMD5(denseRFRegressor), GetMD5(regressor2));
}