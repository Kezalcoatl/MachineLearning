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

TEST(CrossValidationTraining, RegressorTests)
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
	size_t const numThreads = 16;
	const std::string linearKRRRegressorMD5 = "a1f52f0c7439f8f13ccbd56af837ec02";
	const std::string linearKRRDiagnosticsMD5 = "a6e4415cf02c1381f7be04dee5ee20b5";
	const std::string polynomialKRRRegressorMD5 = "5e6afb1374c7393a932a9b6eff8cd658";
	const std::string polynomialKRRDiagnosticsMD5 = "6baac8ddb78fb17845d13b4da170348e";
	const std::string radialBasisKRRRegressorMD5 = "e7dcd8c017147a0e0815687919b84eb2";
	const std::string radialBasisKRRDiagnosticsMD5 = "2cab643f15dac0b9a85dffe7885d9223";
	const std::string sigmoidKRRRegressorMD5 = "bbb5cd62a14f622e4a0d44cdef71d0c7";
	const std::string sigmoidKRRDiagnosticsMD5 = "2fd3b2a2b91fe414cbc286206069c98d";
	const std::string linearSVRRegressorMD5 = "d9a568996852758d5d23c556d2173daf";
	const std::string linearSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	const std::string polynomialSVRRegressorMD5 = "83669b0f6178a085c48bf11ad1a53ffb";
	const std::string polynomialSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	const std::string radialBasisSVRRegressorMD5 = "a974d03f7d33d6d362d8e4bfa4aef889";
	const std::string radialBasisSVRDiagnosticsMD5 = "cd0bcf7cac18bb305ca532dc27e02e0b";
	const std::string sigmoidSVRRegressorMD5 = "92856c41a1c0287d66f698096433f00d";
	const std::string sigmoidSVRDiagnosticsMD5 = "bbf078a8c6648a338fdec7c53ee2e3fe";
	const std::string denseRFRegressorMD5 = "f6f21019febf9fe0455cbcff0ddbaf7d";
	const std::string denseRFDiagnosticsMD5 = "b1f2d546bd20a9834d76731afdf7076d";

	ModifierTypes::NormaliserModifier<T>::CrossValidationTrainingParams normaliserCVParams;
	ModifierTypes::InputPCAModifier<T>::CrossValidationTrainingParams PCACVParams;
	PCACVParams.TargetVarianceToTry = { 0.6, 0.8 };
	ModifierTypes::FeatureSelectionModifier<T>::CrossValidationTrainingParams featureSelectionCVParams;
	featureSelectionCVParams.FeatureFractionsToTry = { 0.7, 0.9 };

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::CrossValidationTrainingParams linearKRRCVParams;
	linearKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	linearKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearKRRDiagnostics, linearKRRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(linearKRRRegressor), linearKRRRegressorMD5);
	EXPECT_EQ(GetMD5(linearKRRDiagnostics), linearKRRDiagnosticsMD5);

	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::CrossValidationTrainingParams polynomialKRRCVParams;
	polynomialKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	polynomialKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.DegreeToTry = { 1.0 };
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, polynomialKRRDiagnostics, polynomialKRRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), polynomialKRRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialKRRDiagnostics), polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::CrossValidationTrainingParams radialBasisKRRCVParams;
	radialBasisKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	radialBasisKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	radialBasisKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, radialBasisKRRDiagnostics, radialBasisKRRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), radialBasisKRRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisKRRDiagnostics), radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::CrossValidationTrainingParams sigmoidKRRCVParams;
	sigmoidKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	sigmoidKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	sigmoidKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	sigmoidKRRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, sigmoidKRRDiagnostics, sigmoidKRRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), sigmoidKRRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidKRRDiagnostics), sigmoidKRRDiagnosticsMD5);

	std::vector<T> linearSVRDiagnostics;
	LinearSVR::CrossValidationTrainingParams linearSVRCVParams;
	linearSVRCVParams.CToTry = { 1.0, 2.0 };
	linearSVRCVParams.EpsilonToTry = { 1.e-3 };
	linearSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	linearSVRCVParams.CacheSizeToTry = { 200 };
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearSVRDiagnostics, linearSVRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(linearSVRRegressor), linearSVRRegressorMD5);
	EXPECT_EQ(GetMD5(linearSVRDiagnostics), linearSVRDiagnosticsMD5);

	std::vector<T> polynomialSVRDiagnostics;
	PolynomialSVR::CrossValidationTrainingParams polynomialSVRCVParams;
	polynomialSVRCVParams.CToTry = { 1.0, 2.0 };
	polynomialSVRCVParams.EpsilonToTry = { 1.e-3 };
	polynomialSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	polynomialSVRCVParams.CacheSizeToTry = { 200 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.DegreeToTry = { 1.0 };
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, polynomialSVRDiagnostics, polynomialSVRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), polynomialSVRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialSVRDiagnostics), polynomialSVRDiagnosticsMD5);

	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::CrossValidationTrainingParams radialBasisSVRCVParams;
	radialBasisSVRCVParams.CToTry = { 1.0, 2.0 };
	radialBasisSVRCVParams.EpsilonToTry = { 1.e-3 };
	radialBasisSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	radialBasisSVRCVParams.CacheSizeToTry = { 200 };
	radialBasisSVRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, radialBasisSVRDiagnostics, radialBasisSVRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(radialBasisSVRRegressor), radialBasisSVRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisSVRDiagnostics), radialBasisSVRDiagnosticsMD5);

	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::CrossValidationTrainingParams sigmoidSVRCVParams;
	sigmoidSVRCVParams.CToTry = { 1.0, 2.0 };
	sigmoidSVRCVParams.EpsilonToTry = { 1.e-3 };
	sigmoidSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	sigmoidSVRCVParams.CacheSizeToTry = { 200 };
	sigmoidSVRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	sigmoidSVRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, sigmoidSVRDiagnostics, sigmoidSVRCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), sigmoidSVRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidSVRDiagnostics), sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::CrossValidationTrainingParams denseRFCVParams;
	denseRFCVParams.NumTreesToTry = { 100, 1000 };
	denseRFCVParams.MinSamplesPerLeafToTry = { 3, 5 };
	denseRFCVParams.SubsamplingFractionToTry = { 1.0 / 3.0, 0.5 };
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, denseRFDiagnostics, denseRFCVParams, normaliserCVParams, PCACVParams, featureSelectionCVParams);
	EXPECT_EQ(GetMD5(denseRFRegressor), denseRFRegressorMD5);
	EXPECT_EQ(GetMD5(denseRFDiagnostics), denseRFDiagnosticsMD5);
}