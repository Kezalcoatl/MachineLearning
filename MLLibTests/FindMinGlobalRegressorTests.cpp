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

TEST(FindMinGlobalTraining, RegressorTests) 
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
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::FourierLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearFourierIRLS;
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::LagrangeLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearLagrangeIRLS;

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
	T const optimisationTolerance = 1.e-2;
	size_t const maxNumCalls = 100;
	size_t const numThreads = 16;
	std::string const linearKRRRegressorMD5 = "bf41fd320889092f85a19bc276fbeba6";
	std::string const linearKRRDiagnosticsMD5 = "25bd173dc2870a9c677d42f0469197c4";
	std::string const polynomialKRRRegressorMD5 = "37f35baf45b8a41d30a0fb397483b6c5";
	std::string const polynomialKRRDiagnosticsMD5 = "4dfd046a12a8b8fa9539b456bb711d86";
	std::string const radialBasisKRRRegressorMD5 = "e5f616123d69399ac9983306ad6e6db5";
	std::string const radialBasisKRRDiagnosticsMD5 = "6f224c7b07faa50ffe47ebd180d46833";
	std::string const sigmoidKRRRegressorMD5 = "9f5ee42b10e81b0cba79f4b3b430bccc";
	std::string const sigmoidKRRDiagnosticsMD5 = "d74f373473f0a1f69a6359a36968a68b";
	std::string const linearSVRRegressorMD5 = "4717f8c05eb4855c68185bd0f8cf6566";
	std::string const linearSVRDiagnosticsMD5 = "30169b4d83ecfbc6d3f3fc8ab39e92be";
	std::string const polynomialSVRRegressorMD5 = "dd6e3d1fc83b81090be81f3fecafa099";
	std::string const polynomialSVRDiagnosticsMD5 = "f8433836ce21b974f40f53d7ef735030";
	std::string const radialBasisSVRRegressorMD5 = "034c0769c878eb3194bbb504f99b5f8f";
	std::string const radialBasisSVRDiagnosticsMD5 = "b1dba4383997a2f6109f6a67f5380079";
	std::string const sigmoidSVRRegressorMD5 = "5e88c20a9c9ec1ab1ae704484befc67c";
	std::string const sigmoidSVRDiagnosticsMD5 = "56eb067604fed3831da1e9acf28b147d";
	std::string const denseRFRegressorMD5 = "d5ed0e92b49e1161dd9a925b482c186d";
	std::string const denseRFDiagnosticsMD5 = "892972ebeab61a813756c62862fc6d30";
	std::string const linearLogitIRLSRegressorMD5 = "";
	std::string const linearLogitIRLSDiagnosticsMD5 = "";
	std::string const linearFourierIRLSRegressorMD5 = "";
	std::string const linearFourierIRLSDiagnosticsMD5 = "";
	std::string const linearLagrangeIRLSRegressorMD5 = "";
	std::string const linearLagrangeIRLSDiagnosticsMD5 = "";

	ModifierTypes::NormaliserModifier<SampleType>::FindMinGlobalTrainingParams normaliserFMGParams;
	ModifierTypes::InputPCAModifier<SampleType>::FindMinGlobalTrainingParams PCAFMGParams;
	PCAFMGParams.LowerTargetVariance = 0.5;
	PCAFMGParams.UpperTargetVariance = 1.0;
	ModifierTypes::FeatureSelectionModifier<SampleType>::FindMinGlobalTrainingParams featureSelectionFMGParams;
	featureSelectionFMGParams.LowerFeatureFraction = 0.5;
	featureSelectionFMGParams.UpperFeatureFraction = 1.0;

	std::vector<T> linearKRRDiagnostics;
	LinearKRR::FindMinGlobalTrainingParams linearKRRFMGParams;
	linearKRRFMGParams.LowerLambda = 1.e-6;
	linearKRRFMGParams.UpperLambda = 10.0;
	linearKRRFMGParams.LowerMaxBasisFunctions = 50;
	linearKRRFMGParams.UpperMaxBasisFunctions = 100;
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearKRRDiagnostics, linearKRRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(linearKRRRegressor), linearKRRRegressorMD5);
	EXPECT_EQ(GetMD5(linearKRRDiagnostics), linearKRRDiagnosticsMD5);
	
	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::FindMinGlobalTrainingParams polynomialKRRFMGParams;
	polynomialKRRFMGParams.LowerLambda = 1.e-6;
	polynomialKRRFMGParams.UpperLambda = 10.0;
	polynomialKRRFMGParams.LowerMaxBasisFunctions = 50;
	polynomialKRRFMGParams.UpperMaxBasisFunctions = 100;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerDegree = 1.0;
	polynomialKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 1.0;
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, polynomialKRRDiagnostics, polynomialKRRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), polynomialKRRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialKRRDiagnostics), polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::FindMinGlobalTrainingParams radialBasisKRRFMGParams;
	radialBasisKRRFMGParams.LowerLambda = 1.e-6;
	radialBasisKRRFMGParams.UpperLambda = 10.0;
	radialBasisKRRFMGParams.LowerMaxBasisFunctions = 50;
	radialBasisKRRFMGParams.UpperMaxBasisFunctions = 100;
	radialBasisKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	radialBasisKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, radialBasisKRRDiagnostics, radialBasisKRRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), radialBasisKRRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisKRRDiagnostics), radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::FindMinGlobalTrainingParams sigmoidKRRFMGParams;
	sigmoidKRRFMGParams.LowerLambda = 1.e-6;
	sigmoidKRRFMGParams.UpperLambda = 10.0;
	sigmoidKRRFMGParams.LowerMaxBasisFunctions = 50;
	sigmoidKRRFMGParams.UpperMaxBasisFunctions = 100;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerGamma = 1.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperGamma = 2.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.LowerCoeff = 0.0;
	sigmoidKRRFMGParams.KernelFindMinGlobalTrainingParams.UpperCoeff = 1.0;
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, sigmoidKRRDiagnostics, sigmoidKRRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), sigmoidKRRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidKRRDiagnostics), sigmoidKRRDiagnosticsMD5);

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
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearSVRDiagnostics, linearSVRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(linearSVRRegressor), linearSVRRegressorMD5);
	EXPECT_EQ(GetMD5(linearSVRDiagnostics), linearSVRDiagnosticsMD5);

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
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, polynomialSVRDiagnostics, polynomialSVRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), polynomialSVRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialSVRDiagnostics), polynomialSVRDiagnosticsMD5);

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
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, radialBasisSVRDiagnostics, radialBasisSVRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(radialBasisSVRRegressor), radialBasisSVRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisSVRDiagnostics), radialBasisSVRDiagnosticsMD5);

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
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, sigmoidSVRDiagnostics, sigmoidSVRFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), sigmoidSVRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidSVRDiagnostics), sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::FindMinGlobalTrainingParams denseRFFMGParams;
	denseRFFMGParams.LowerNumTrees = 50;
	denseRFFMGParams.UpperNumTrees = 200;
	denseRFFMGParams.LowerMinSamplesPerLeaf = 2;
	denseRFFMGParams.UpperMinSamplesPerLeaf = 10;
	denseRFFMGParams.LowerSubsamplingFraction = 0.2;
	denseRFFMGParams.UpperSubsamplingFraction = 0.8;
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, denseRFDiagnostics, denseRFFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(denseRFRegressor), denseRFRegressorMD5);
	EXPECT_EQ(GetMD5(denseRFDiagnostics), denseRFDiagnosticsMD5);

	std::vector<T> linearFourierIRLSDiagnostics;
	LinearFourierIRLS::FindMinGlobalTrainingParams linearFourierIRLSFMGParams;
	denseRFFMGParams.LowerNumTrees = 50;
	denseRFFMGParams.UpperNumTrees = 200;
	denseRFFMGParams.LowerMinSamplesPerLeaf = 2;
	denseRFFMGParams.UpperMinSamplesPerLeaf = 10;
	denseRFFMGParams.LowerSubsamplingFraction = 0.2;
	denseRFFMGParams.UpperSubsamplingFraction = 0.8;
	auto const linearFourierIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearFourierIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearFourierIRLSDiagnostics, linearFourierIRLSFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(linearFourierIRLSRegressor), linearFourierIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearFourierIRLSDiagnostics), linearFourierIRLSDiagnosticsMD5);

	std::vector<T> linearLagrangeIRLSDiagnostics;
	LinearLagrangeIRLS::FindMinGlobalTrainingParams linearLagrangeIRLSFMGParams;
	denseRFFMGParams.LowerNumTrees = 50;
	denseRFFMGParams.UpperNumTrees = 200;
	denseRFFMGParams.LowerMinSamplesPerLeaf = 2;
	denseRFFMGParams.UpperMinSamplesPerLeaf = 10;
	denseRFFMGParams.LowerSubsamplingFraction = 0.2;
	denseRFFMGParams.UpperSubsamplingFraction = 0.8;
	auto const linearLagrangeIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorFindMinGlobal<LinearLagrangeIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, optimisationTolerance, numThreads, maxNumCalls, linearLagrangeIRLSDiagnostics, linearLagrangeIRLSFMGParams, normaliserFMGParams, featureSelectionFMGParams, PCAFMGParams);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSRegressor), linearLagrangeIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSDiagnostics), linearLagrangeIRLSDiagnosticsMD5);
}