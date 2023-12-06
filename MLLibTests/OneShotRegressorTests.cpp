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

TEST(OneShotTraining, RegressorTests)
{
	using namespace Regressors;
	typedef col_vector<double> SampleType;
	typedef SampleType::type T;
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
	CrossValidationMetric const metric = CrossValidationMetric::SumSquareMean;
	size_t const numFolds = 4;
	size_t const maxNumCalls = 1000;
	size_t const numThreads = 16;
	std::string const linearKRRRegressorMD5 = "3ee061dc9c21c0fb86dd372b887ebf2b";
	std::string const linearKRRDiagnosticsMD5 = "b3fea5b0c03d4aafb817e7f38272bf0f";
	std::string const polynomialKRRRegressorMD5 = "6195121f14edef73c5356df546dcb1d0";
	std::string const polynomialKRRDiagnosticsMD5 = "b3fea5b0c03d4aafb817e7f38272bf0f";
	std::string const radialBasisKRRRegressorMD5 = "a3cef6e7f2ab31344ce478011f8e23da";
	std::string const radialBasisKRRDiagnosticsMD5 = "152510c1200a9c21bdbd80e02ece6918";
	std::string const sigmoidKRRRegressorMD5 = "6f922da169106633872f790a27119d65";
	std::string const sigmoidKRRDiagnosticsMD5 = "67bd96ab0cead9ad0706eec832a6d7a5";
	std::string const linearSVRRegressorMD5 = "9006d04a29e99b41dcfd31b61888fd69";
	std::string const linearSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	std::string const polynomialSVRRegressorMD5 = "94512b091cdb7b284e5039a0aeddaeee";
	std::string const polynomialSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	std::string const radialBasisSVRRegressorMD5 = "04259b503902a942a551677461bd03bd";
	std::string const radialBasisSVRDiagnosticsMD5 = "ab52eb8ef6aaeb0acc2f0f471986b314";
	std::string const sigmoidSVRRegressorMD5 = "79cdc6eb1551e0c360ab05676d70d0d2";
	std::string const sigmoidSVRDiagnosticsMD5 = "a119c2879cbbf92c29575d8e0518bd63";
	std::string const denseRFRegressorMD5 = "35b8b7c53a53f3450317714a4e090300";
	std::string const denseRFDiagnosticsMD5 = "8dce9be578d78c26ac25bb27fed929bd";
	/*
	ModifierTypes::NormaliserModifier<SampleType>::OneShotTrainingParams normaliserOSParams;
	ModifierTypes::InputPCAModifier<SampleType>::OneShotTrainingParams PCAOSParams;
	PCAOSParams.TargetVariance = 0.9;
	ModifierTypes::FeatureSelectionModifier<SampleType>::OneShotTrainingParams featureSelectionOSParams;
	featureSelectionOSParams.FeatureFraction = 0.7;

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::OneShotTrainingParams linearKRROSParams;
	linearKRROSParams.MaxBasisFunctions = 400;
	linearKRROSParams.Lambda = 1.e-6;
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearKRRDiagnostics, linearKRROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(linearKRRRegressor), linearKRRRegressorMD5);
	EXPECT_EQ(GetMD5(linearKRRDiagnostics), linearKRRDiagnosticsMD5);

	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::OneShotTrainingParams polynomialKRROSParams;
	polynomialKRROSParams.MaxBasisFunctions = 400;
	polynomialKRROSParams.Lambda = 1.e-6;
	polynomialKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialKRRDiagnostics, polynomialKRROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), polynomialKRRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialKRRDiagnostics), polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::OneShotTrainingParams radialBasisKRROSParams;
	radialBasisKRROSParams.MaxBasisFunctions = 400;
	radialBasisKRROSParams.Lambda = 1e-6;
	radialBasisKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisKRRDiagnostics, radialBasisKRROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), radialBasisKRRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisKRRDiagnostics), radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::OneShotTrainingParams sigmoidKRROSParams;
	sigmoidKRROSParams.MaxBasisFunctions = 400;
	sigmoidKRROSParams.Lambda = 1.e-6;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidKRRDiagnostics, sigmoidKRROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), sigmoidKRRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidKRRDiagnostics), sigmoidKRRDiagnosticsMD5);

	std::vector<T> linearSVRDiagnostics;
	LinearSVR::OneShotTrainingParams linearSVROSParams;
	linearSVROSParams.C = 1.0;
	linearSVROSParams.Epsilon = 1.e-3;
	linearSVROSParams.EpsilonInsensitivity = 0.1;
	linearSVROSParams.CacheSize = 200;
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearSVRDiagnostics, linearSVROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(linearSVRRegressor), linearSVRRegressorMD5);
	EXPECT_EQ(GetMD5(linearSVRDiagnostics), linearSVRDiagnosticsMD5);

	std::vector<T> polynomialSVRDiagnostics;
	PolynomialSVR::OneShotTrainingParams polynomialSVROSParams;
	polynomialSVROSParams.C = 1.0;
	polynomialSVROSParams.Epsilon = 1.e-3;
	polynomialSVROSParams.EpsilonInsensitivity = 0.1;
	polynomialSVROSParams.CacheSize = 200;
	polynomialSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialSVROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialSVRDiagnostics, polynomialSVROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), polynomialSVRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialSVRDiagnostics), polynomialSVRDiagnosticsMD5);

	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::OneShotTrainingParams radialBasisSVROSParams;
	radialBasisSVROSParams.C = 1.0;
	radialBasisSVROSParams.Epsilon = 1.e-3;
	radialBasisSVROSParams.EpsilonInsensitivity = 0.1;
	radialBasisSVROSParams.CacheSize = 200;
	radialBasisSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisSVRDiagnostics, radialBasisSVROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(radialBasisSVRRegressor), radialBasisSVRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisSVRDiagnostics), radialBasisSVRDiagnosticsMD5);

	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::OneShotTrainingParams sigmoidSVROSParams;
	sigmoidSVROSParams.C = 1.0;
	sigmoidSVROSParams.Epsilon = 1.e-3;
	sigmoidSVROSParams.EpsilonInsensitivity = 0.1;
	sigmoidSVROSParams.CacheSize = 200;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidSVRDiagnostics, sigmoidSVROSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), sigmoidSVRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidSVRDiagnostics), sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::OneShotTrainingParams denseRFOSParams;
	denseRFOSParams.NumTrees = 1000;
	denseRFOSParams.MinSamplesPerLeaf = 5;
	denseRFOSParams.SubsamplingFraction = 1.0 / 3.0;
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, denseRFDiagnostics, denseRFOSParams, normaliserOSParams, featureSelectionOSParams, PCAOSParams);
	EXPECT_EQ(GetMD5(denseRFRegressor), denseRFRegressorMD5);
	EXPECT_EQ(GetMD5(denseRFDiagnostics), denseRFDiagnosticsMD5);
	*/
}