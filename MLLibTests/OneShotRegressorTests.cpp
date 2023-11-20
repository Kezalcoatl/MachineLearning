#include "gtest/gtest.h"
#include <MLLib/Regressor.h>
#include <dlib/md5.h>

TEST(OneShotTraining, RegressorTests)
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
	std::stringstream regressor;
	std::stringstream diagnostics;
	const std::string linearKRRRegressorMD5 = "3ee061dc9c21c0fb86dd372b887ebf2b";
	const std::string linearKRRDiagnosticsMD5 = "b3fea5b0c03d4aafb817e7f38272bf0f";
	const std::string polynomialKRRRegressorMD5 = "6195121f14edef73c5356df546dcb1d0";
	const std::string polynomialKRRDiagnosticsMD5 = "b3fea5b0c03d4aafb817e7f38272bf0f";
	const std::string radialBasisKRRRegressorMD5 = "a3cef6e7f2ab31344ce478011f8e23da";
	const std::string radialBasisKRRDiagnosticsMD5 = "152510c1200a9c21bdbd80e02ece6918";
	const std::string sigmoidKRRRegressorMD5 = "6f922da169106633872f790a27119d65";
	const std::string sigmoidKRRDiagnosticsMD5 = "67bd96ab0cead9ad0706eec832a6d7a5";
	const std::string linearSVRRegressorMD5 = "9006d04a29e99b41dcfd31b61888fd69";
	const std::string linearSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	const std::string polynomialSVRRegressorMD5 = "94512b091cdb7b284e5039a0aeddaeee";
	const std::string polynomialSVRDiagnosticsMD5 = "c10726199aa5d8d78a60d962e62fbaf9";
	const std::string radialBasisSVRRegressorMD5 = "04259b503902a942a551677461bd03bd";
	const std::string radialBasisSVRDiagnosticsMD5 = "ab52eb8ef6aaeb0acc2f0f471986b314";
	const std::string sigmoidSVRRegressorMD5 = "79cdc6eb1551e0c360ab05676d70d0d2";
	const std::string sigmoidSVRDiagnosticsMD5 = "a119c2879cbbf92c29575d8e0518bd63";
	const std::string denseRFRegressorMD5 = "35b8b7c53a53f3450317714a4e090300";
	const std::string denseRFDiagnosticsMD5 = "8dce9be578d78c26ac25bb27fed929bd";
	std::string regressorMD5;
	std::string diagnosticsMD5;

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
	regressor.clear();
	diagnostics.clear();
	serialize(linearKRRRegressor, regressor);
	dlib::serialize(linearKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, linearKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, linearKRRDiagnosticsMD5);

	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::OneShotTrainingParams polynomialKRROSParams;
	polynomialKRROSParams.MaxBasisFunctions = 400;
	polynomialKRROSParams.Lambda = 1.e-6;
	polynomialKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	polynomialKRROSParams.KernelOneShotTrainingParams.Degree = 1.0;
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, polynomialKRRDiagnostics, polynomialKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(polynomialKRRRegressor, regressor);
	dlib::serialize(polynomialKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, polynomialKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::OneShotTrainingParams radialBasisKRROSParams;
	radialBasisKRROSParams.MaxBasisFunctions = 400;
	radialBasisKRROSParams.Lambda = 1e-6;
	radialBasisKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisKRRDiagnostics, radialBasisKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(radialBasisKRRRegressor, regressor);
	dlib::serialize(radialBasisKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, radialBasisKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::OneShotTrainingParams sigmoidKRROSParams;
	sigmoidKRROSParams.MaxBasisFunctions = 400;
	sigmoidKRROSParams.Lambda = 1.e-6;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidKRROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidKRRDiagnostics, sigmoidKRROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(sigmoidKRRRegressor, regressor);
	dlib::serialize(sigmoidKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, sigmoidKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, sigmoidKRRDiagnosticsMD5);

	std::vector<T> linearSVRDiagnostics;
	LinearSVR::OneShotTrainingParams linearSVROSParams;
	linearSVROSParams.C = 1.0;
	linearSVROSParams.Epsilon = 1.e-3;
	linearSVROSParams.EpsilonInsensitivity = 0.1;
	linearSVROSParams.CacheSize = 200;
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearSVRDiagnostics, linearSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(linearSVRRegressor, regressor);
	dlib::serialize(linearSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, linearSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, linearSVRDiagnosticsMD5);

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
	regressor.clear();
	diagnostics.clear();
	serialize(polynomialSVRRegressor, regressor);
	dlib::serialize(polynomialSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, polynomialSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, polynomialSVRDiagnosticsMD5);

	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::OneShotTrainingParams radialBasisSVROSParams;
	radialBasisSVROSParams.C = 1.0;
	radialBasisSVROSParams.Epsilon = 1.e-3;
	radialBasisSVROSParams.EpsilonInsensitivity = 0.1;
	radialBasisSVROSParams.CacheSize = 200;
	radialBasisSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, radialBasisSVRDiagnostics, radialBasisSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(radialBasisSVRRegressor, regressor);
	dlib::serialize(radialBasisSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, radialBasisSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, radialBasisSVRDiagnosticsMD5);

	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::OneShotTrainingParams sigmoidSVROSParams;
	sigmoidSVROSParams.C = 1.0;
	sigmoidSVROSParams.Epsilon = 1.e-3;
	sigmoidSVROSParams.EpsilonInsensitivity = 0.1;
	sigmoidSVROSParams.CacheSize = 200;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Gamma = 1.0;
	sigmoidSVROSParams.KernelOneShotTrainingParams.Coeff = 0.0;
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, sigmoidSVRDiagnostics, sigmoidSVROSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(sigmoidSVRRegressor, regressor);
	dlib::serialize(sigmoidSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, sigmoidSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::OneShotTrainingParams denseRFOSParams;
	denseRFOSParams.NumTrees = 1000;
	denseRFOSParams.MinSamplesPerLeaf = 5;
	denseRFOSParams.SubsamplingFraction = 1.0 / 3.0;
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, denseRFDiagnostics, denseRFOSParams, normaliserOSParams, PCAOSParams, featureSelectionOSParams);
	regressor.clear();
	diagnostics.clear();
	serialize(denseRFRegressor, regressor);
	dlib::serialize(denseRFDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, denseRFRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, denseRFDiagnosticsMD5);
}