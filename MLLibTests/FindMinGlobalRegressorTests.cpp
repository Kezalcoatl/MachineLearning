#include "gtest/gtest.h"
#include <MLLib/Regressor.h>
#include <dlib/md5.h>

TEST(FindMinGlobalTraining, RegressorTests) 
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

	static size_t const numExamples = 100;
	static size_t const numOrdinates = 50;

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
	T const optimisationTolerance = 1.e-6;
	size_t const maxNumCalls = 1000;
	size_t const numThreads = 16;
	std::stringstream regressor;
	std::stringstream diagnostics;
	const std::string linearKRRRegressorMD5 = "cf2704c7eb542b7ca37b9c7579173816";
	const std::string linearKRRDiagnosticsMD5 = "ad8bc9aa7681fdb592c2af07bdb6bf69";
	const std::string polynomialKRRRegressorMD5 = "5b4606f5199c358d07466b3f2c36b5b7";
	const std::string polynomialKRRDiagnosticsMD5 = "eb2164e6e471cfd109de8f43070baef4";
	const std::string radialBasisKRRRegressorMD5 = "4c6d6c182e99249492ed28bea83b9ef6";
	const std::string radialBasisKRRDiagnosticsMD5 = "30a273405ae37289ac2bd45f42d3b70b";
	const std::string sigmoidKRRRegressorMD5 = "3f00294581dc4c7435109b92a7f33783";
	const std::string sigmoidKRRDiagnosticsMD5 = "ecbfcdd7e27732ff942916f6e40c71a7";
	const std::string linearSVRRegressorMD5 = "09596c22b1edcf0cbb6a0820c9969c11";
	const std::string linearSVRDiagnosticsMD5 = "438e3c3c6e42451d4ff0c28a3e91a6d6";
	const std::string polynomialSVRRegressorMD5 = "dcc330f205a61688c1b799f9ac889131";
	const std::string polynomialSVRDiagnosticsMD5 = "b66b9405c4fe7c287b6df86e9d851aea";
	const std::string radialBasisSVRRegressorMD5 = "31517800a1e9b03f1318d090eddcf10e";
	const std::string radialBasisSVRDiagnosticsMD5 = "7ac0b1de843c6c54e82798b56252da6d";
	const std::string sigmoidSVRRegressorMD5 = "c3adc56f74e5779bdd8bd8e751e91585";
	const std::string sigmoidSVRDiagnosticsMD5 = "b2b768bf60e7dc4bac708a549e3ec97c";
	const std::string denseRFRegressorMD5 = "ba75775ef04625683f6d26a23ba27cfb";
	const std::string denseRFDiagnosticsMD5 = "70600da8e1aa29238dbde3b57921ebae";
	std::string regressorMD5;
	std::string diagnosticsMD5;

	ModifierTypes::NormaliserModifier<T>::FindMinGlobalTrainingParams normaliserFMGParams;
	ModifierTypes::InputPCAModifier<T>::FindMinGlobalTrainingParams PCAFMGParams;
	PCAFMGParams.LowerTargetVariance = 0.5;
	PCAFMGParams.UpperTargetVariance = 1.0;
	ModifierTypes::FeatureSelectionModifier<T>::FindMinGlobalTrainingParams featureSelectionFMGParams;
	featureSelectionFMGParams.LowerFeatureFraction = 0.2;
	featureSelectionFMGParams.UpperFeatureFraction = 1.0;

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::FindMinGlobalTrainingParams linearKRRFMGParams;
	linearKRRFMGParams.LowerLambda = 1.e-6;
	linearKRRFMGParams.UpperLambda = 10.0;
	linearKRRFMGParams.LowerMaxBasisFunctions = 50;
	linearKRRFMGParams.UpperMaxBasisFunctions = 500;
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearKRRDiagnostics, linearKRRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(linearKRRRegressor, regressor);
	dlib::serialize(linearKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, linearKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, linearKRRDiagnosticsMD5);

	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::FindMinGlobalTrainingParams polynomialKRRFMGParams;
	polynomialKRRFMGParams.LowerLambda = 1.e-6;
	polynomialKRRFMGParams.UpperLambda = 10.0;
	polynomialKRRFMGParams.LowerMaxBasisFunctions = 50;
	polynomialKRRFMGParams.UpperMaxBasisFunctions = 500;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerDegree = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 1.0;
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, polynomialKRRDiagnostics, polynomialKRRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(polynomialKRRRegressor, regressor);
	dlib::serialize(polynomialKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, polynomialKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::FindMinGlobalTrainingParams radialBasisKRRFMGParams;
	radialBasisKRRFMGParams.LowerLambda = 1.e-6;
	radialBasisKRRFMGParams.UpperLambda = 10.0;
	radialBasisKRRFMGParams.LowerMaxBasisFunctions = 50;
	radialBasisKRRFMGParams.UpperMaxBasisFunctions = 500;
	radialBasisKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	radialBasisKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, radialBasisKRRDiagnostics, radialBasisKRRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(radialBasisKRRRegressor, regressor);
	dlib::serialize(radialBasisKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, radialBasisKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::FindMinGlobalTrainingParams sigmoidKRRFMGParams;
	sigmoidKRRFMGParams.LowerLambda = 1.e-6;
	sigmoidKRRFMGParams.UpperLambda = 10.0;
	sigmoidKRRFMGParams.LowerMaxBasisFunctions = 50;
	sigmoidKRRFMGParams.UpperMaxBasisFunctions = 500;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, sigmoidKRRDiagnostics, sigmoidKRRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(sigmoidKRRRegressor, regressor);
	dlib::serialize(sigmoidKRRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, sigmoidKRRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, sigmoidKRRDiagnosticsMD5);

	std::vector<T> linearSVRDiagnostics;
	LinearSVR::FindMinGlobalTrainingParams linearSVRFMGParams;
	linearSVRFMGParams.LowerC = 1.0;
	linearSVRFMGParams.UpperC = 2.0;
	linearSVRFMGParams.LowerEpsilon = 1e-4;
	linearSVRFMGParams.UpperEpsilon = 0.1;
	linearSVRFMGParams.LowerEpsilonInsensitivity = 1e-2;
	linearSVRFMGParams.UpperEpsilonInsensitivity = 1.0;
	linearSVRFMGParams.LowerCacheSize = 50;
	linearSVRFMGParams.UpperCacheSize = 300;
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearSVRDiagnostics, linearSVRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(linearSVRRegressor, regressor);
	dlib::serialize(linearSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, linearSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, linearSVRDiagnosticsMD5);

	std::vector<T> polynomialSVRDiagnostics;
	PolynomialSVR::FindMinGlobalTrainingParams polynomialSVRFMGParams;
	polynomialSVRFMGParams.LowerC = 1.0;
	polynomialSVRFMGParams.UpperC = 2.0;
	polynomialSVRFMGParams.LowerEpsilon = 1e-4;
	polynomialSVRFMGParams.UpperEpsilon = 0.1;
	polynomialSVRFMGParams.LowerEpsilonInsensitivity = 1e-2;
	polynomialSVRFMGParams.UpperEpsilonInsensitivity = 1.0;
	polynomialSVRFMGParams.LowerCacheSize = 50;
	polynomialSVRFMGParams.UpperCacheSize = 300;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerDegree = 1.0;
	polynomialSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 1.0;
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, polynomialSVRDiagnostics, polynomialSVRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(polynomialSVRRegressor, regressor);
	dlib::serialize(polynomialSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, polynomialSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, polynomialSVRDiagnosticsMD5);

	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::FindMinGlobalTrainingParams radialBasisSVRFMGParams;
	radialBasisSVRFMGParams.LowerC = 1.0;
	radialBasisSVRFMGParams.UpperC = 2.0;
	radialBasisSVRFMGParams.LowerEpsilon = 1e-4;
	radialBasisSVRFMGParams.UpperEpsilon = 0.1;
	radialBasisSVRFMGParams.LowerEpsilonInsensitivity = 1e-2;
	radialBasisSVRFMGParams.UpperEpsilonInsensitivity = 1.0;
	radialBasisSVRFMGParams.LowerCacheSize = 50;
	radialBasisSVRFMGParams.UpperCacheSize = 300;
	radialBasisSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	radialBasisSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, radialBasisSVRDiagnostics, radialBasisSVRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(radialBasisSVRRegressor, regressor);
	dlib::serialize(radialBasisSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, radialBasisSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, radialBasisSVRDiagnosticsMD5);

	std::vector<T> sigmoidSVRDiagnostics;
	SigmoidSVR::FindMinGlobalTrainingParams sigmoidSVRFMGParams;
	sigmoidSVRFMGParams.LowerC = 1.0;
	sigmoidSVRFMGParams.UpperC = 2.0;
	sigmoidSVRFMGParams.LowerEpsilon = 1e-4;
	sigmoidSVRFMGParams.UpperEpsilon = 0.1;
	sigmoidSVRFMGParams.LowerEpsilonInsensitivity = 1e-2;
	sigmoidSVRFMGParams.UpperEpsilonInsensitivity = 1.0;
	sigmoidSVRFMGParams.LowerCacheSize = 50;
	sigmoidSVRFMGParams.UpperCacheSize = 300;
	sigmoidSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	sigmoidSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	sigmoidSVRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	sigmoidSVRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, sigmoidSVRDiagnostics, sigmoidSVRFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(sigmoidSVRRegressor, regressor);
	dlib::serialize(sigmoidSVRDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, sigmoidSVRRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::FindMinGlobalTrainingParams denseRFFMGParams;
	denseRFFMGParams.LowerNumTrees = 50;
	denseRFFMGParams.UpperNumTrees = 200;
	denseRFFMGParams.LowerMinSamplesPerLeaf = 2;
	denseRFFMGParams.UpperMinSamplesPerLeaf = 10;
	denseRFFMGParams.LowerSubsamplingFraction = 0.2;
	denseRFFMGParams.UpperSubsamplingFraction = 0.8;
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, denseRFDiagnostics, denseRFFMGParams, normaliserFMGParams, PCAFMGParams, featureSelectionFMGParams);
	regressor.clear();
	diagnostics.clear();
	serialize(denseRFRegressor, regressor);
	dlib::serialize(denseRFDiagnostics, diagnostics);
	regressorMD5 = dlib::md5(regressor);
	diagnosticsMD5 = dlib::md5(diagnostics);
	EXPECT_EQ(regressorMD5, denseRFRegressorMD5);
	EXPECT_EQ(diagnosticsMD5, denseRFDiagnosticsMD5);
}