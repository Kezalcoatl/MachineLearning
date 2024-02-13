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
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::LogitLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearLogitIRLS;

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
	std::stringstream regressorSS;
	
	ModifierTypes::NormaliserModifier<SampleType>::OneShotTrainingParams normaliserOSParams;
	ModifierTypes::InputPCAModifier<SampleType>::OneShotTrainingParams PCAOSParams;
	PCAOSParams.TargetVariance = 0.9;
	ModifierTypes::FeatureSelectionModifier<SampleType>::OneShotTrainingParams featureSelectionOSParams;
	featureSelectionOSParams.FeatureFraction = 0.7;

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::OneShotTrainingParams linearKRROSParams;
	linearKRROSParams.MaxBasisFunctions = 400;
	linearKRROSParams.Lambda = 1.e-6;
	auto linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearKRRDiagnostics, linearKRROSParams, normaliserOSParams);
	regressorSS.clear();
	serialize(linearKRRRegressor, regressorSS);
	decltype(linearKRRRegressor) linearKRRRegressor2;
	deserialize(linearKRRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(linearKRRRegressor), GetMD5(linearKRRRegressor2));

	regressorSS.clear();
	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::OneShotTrainingParams polynomialKRROSParams;
	polynomialKRROSParams.MaxBasisFunctions = 400;
	polynomialKRROSParams.Lambda = 1.e-6;
	polynomialKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialKRRDiagnostics, polynomialKRROSParams, normaliserOSParams);
	serialize(polynomialKRRRegressor, regressorSS);
	decltype(polynomialKRRRegressor) polynomialKRRRegressor2;
	deserialize(polynomialKRRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), GetMD5(polynomialKRRRegressor2));

	regressorSS.clear();
	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::OneShotTrainingParams radialBasisKRROSParams;
	radialBasisKRROSParams.MaxBasisFunctions = 400;
	radialBasisKRROSParams.Lambda = 1e-6;
	radialBasisKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisKRRDiagnostics, radialBasisKRROSParams, normaliserOSParams);
	serialize(radialBasisKRRRegressor, regressorSS);
	decltype(radialBasisKRRRegressor) radialBasisKRRRegressor2;
	deserialize(radialBasisKRRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), GetMD5(radialBasisKRRRegressor2));

	regressorSS.clear();
	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::OneShotTrainingParams sigmoidKRROSParams;
	sigmoidKRROSParams.MaxBasisFunctions = 400;
	sigmoidKRROSParams.Lambda = 1.e-6;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidKRRDiagnostics, sigmoidKRROSParams, normaliserOSParams);
	serialize(sigmoidKRRRegressor, regressorSS);
	decltype(sigmoidKRRRegressor) sigmoidKRRRegressor2;
	deserialize(sigmoidKRRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), GetMD5(sigmoidKRRRegressor2));

	regressorSS.clear();
	std::vector<T> linearSVRDiagnostics;
	LinearSVR::OneShotTrainingParams linearSVROSParams;
	linearSVROSParams.C = 1.0;
	linearSVROSParams.Epsilon = 1.e-3;
	linearSVROSParams.EpsilonInsensitivity = 0.1;
	linearSVROSParams.CacheSize = 200;
	auto linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearSVRDiagnostics, linearSVROSParams, normaliserOSParams);
	serialize(linearSVRRegressor, regressorSS);
	decltype(linearSVRRegressor) linearSVRRegressor2;
	deserialize(linearSVRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(linearSVRRegressor), GetMD5(linearSVRRegressor2));

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
	auto polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialSVRDiagnostics, polynomialSVROSParams, normaliserOSParams);
	serialize(polynomialSVRRegressor, regressorSS);
	decltype(polynomialSVRRegressor) polynomialSVRRegressor2;
	deserialize(polynomialSVRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), GetMD5(polynomialSVRRegressor2));

	regressorSS.clear();
	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::OneShotTrainingParams radialBasisSVROSParams;
	radialBasisSVROSParams.C = 1.0;
	radialBasisSVROSParams.Epsilon = 1.e-3;
	radialBasisSVROSParams.EpsilonInsensitivity = 0.1;
	radialBasisSVROSParams.CacheSize = 200;
	radialBasisSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisSVRDiagnostics, radialBasisSVROSParams, normaliserOSParams);
	serialize(radialBasisSVRRegressor, regressorSS);
	decltype(radialBasisSVRRegressor) radialBasisSVRRegressor2;
	deserialize(radialBasisSVRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(radialBasisSVRRegressor), GetMD5(radialBasisSVRRegressor2));

	regressorSS.clear();
	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::OneShotTrainingParams sigmoidSVROSParams;
	sigmoidSVROSParams.C = 1.0;
	sigmoidSVROSParams.Epsilon = 1.e-3;
	sigmoidSVROSParams.EpsilonInsensitivity = 0.1;
	sigmoidSVROSParams.CacheSize = 200;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidSVRDiagnostics, sigmoidSVROSParams, normaliserOSParams);
	serialize(sigmoidSVRRegressor, regressorSS);
	decltype(sigmoidSVRRegressor) sigmoidSVRRegressor2;
	deserialize(sigmoidSVRRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), GetMD5(sigmoidSVRRegressor2));

	regressorSS.clear();
	std::vector<T> denseRFDiagnostics;
	DenseRF::OneShotTrainingParams denseRFOSParams;
	denseRFOSParams.NumTrees = 1000;
	denseRFOSParams.MinSamplesPerLeaf = 5;
	denseRFOSParams.SubsamplingFraction = 1.0 / 3.0;
	auto denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, denseRFDiagnostics, denseRFOSParams, normaliserOSParams);
	serialize(denseRFRegressor, regressorSS);
	decltype(denseRFRegressor) denseRFRegressor2;
	deserialize(denseRFRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(denseRFRegressor), GetMD5(denseRFRegressor2));

	regressorSS.clear();
	std::vector<T> linearLogitIRLSDiagnostics;
	LinearLogitIRLS::OneShotTrainingParams linearLogitIRLSOSParams;
	linearLogitIRLSOSParams.ConvergenceTolerance = 1e-4;
	linearLogitIRLSOSParams.Lambda = 0.1;
	linearLogitIRLSOSParams.MaxBasisFunctions = 50;
	linearLogitIRLSOSParams.MaxNumIterations = 100;
	auto linearLogitIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearLogitIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearLogitIRLSDiagnostics, linearLogitIRLSOSParams, normaliserOSParams);
	serialize(linearLogitIRLSRegressor, regressorSS);
	decltype(linearLogitIRLSRegressor) linearLogitIRLSRegressor2;
	deserialize(linearLogitIRLSRegressor2, regressorSS);
	EXPECT_EQ(GetMD5(linearLogitIRLSRegressor), GetMD5(linearLogitIRLSRegressor2));

}