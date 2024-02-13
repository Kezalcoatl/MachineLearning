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
	size_t const numThreads = 16;
	std::string const linearKRRRegressorMD5 = "01d0975dad2d88b3908e9c5bbc1379f1";
	std::string const linearKRRDiagnosticsMD5 = "1ff023f4a6c549cda0083397d0986831";
	std::string const polynomialKRRRegressorMD5 = "7a5d43adef3987ec9c55c06c53d03cfd";
	std::string const polynomialKRRDiagnosticsMD5 = "a8879cb3594d530e4a17a38f02e1e225";
	std::string const radialBasisKRRRegressorMD5 = "09b8634a60534767eb8bfdf411468707";
	std::string const radialBasisKRRDiagnosticsMD5 = "3c2246bd20b4f201de45a28a3a3293ed";
	std::string const sigmoidKRRRegressorMD5 = "ddaa8272963ba8c01c3fcc3a1772b313";
	std::string const sigmoidKRRDiagnosticsMD5 = "ab6f88d27e0c4702adae2a1e3aed53ab";
	std::string const linearSVRRegressorMD5 = "cb9986cedb0f358d664c1f7951b477df";
	std::string const linearSVRDiagnosticsMD5 = "5754c786a1044105c0cbf1e4b430a72a";
	std::string const polynomialSVRRegressorMD5 = "5006fd5bc68f112a5facea93f9ed3468";
	std::string const polynomialSVRDiagnosticsMD5 = "5754c786a1044105c0cbf1e4b430a72a";
	std::string const radialBasisSVRRegressorMD5 = "e449ef20f69f2e081e7335237fd33292";
	std::string const radialBasisSVRDiagnosticsMD5 = "e4808470c2827efa985013ed42ac9f44";
	std::string const sigmoidSVRRegressorMD5 = "3ab03644c67fc2bdaef1668d23d0aaf9";
	std::string const sigmoidSVRDiagnosticsMD5 = "ae87e39274f21156ccf445b0f4491099";
	std::string const denseRFRegressorMD5 = "832556f88cb9acece4dd3c377d8ecc84";
	std::string const denseRFDiagnosticsMD5 = "b1f2d546bd20a9834d76731afdf7076d";
	std::string const linearLogitIRLSRegressorMD5 = "";
	std::string const linearLogitIRLSDiagnosticsMD5 = "";
	std::string const linearFourierIRLSRegressorMD5 = "";
	std::string const linearFourierIRLSDiagnosticsMD5 = "";
	std::string const linearLagrangeIRLSRegressorMD5 = "";
	std::string const linearLagrangeIRLSDiagnosticsMD5 = "";

	ModifierTypes::NormaliserModifier<SampleType>::CrossValidationTrainingParams normaliserCVParams;
	ModifierTypes::InputPCAModifier<SampleType>::CrossValidationTrainingParams PCACVParams;
	PCACVParams.TargetVarianceToTry = { 0.6, 0.8 };
	ModifierTypes::FeatureSelectionModifier<SampleType>::CrossValidationTrainingParams featureSelectionCVParams;
	featureSelectionCVParams.FeatureFractionsToTry = { 0.7, 0.9 };
	
	std::vector<T> linearKRRDiagnostics;
	LinearKRR::CrossValidationTrainingParams linearKRRCVParams;
	linearKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	linearKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	auto const linearKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearKRRDiagnostics, linearKRRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(linearKRRRegressor), linearKRRRegressorMD5);
	EXPECT_EQ(GetMD5(linearKRRDiagnostics), linearKRRDiagnosticsMD5);

	std::vector<T> polynomialKRRDiagnostics;
	PolynomialKRR::CrossValidationTrainingParams polynomialKRRCVParams;
	polynomialKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	polynomialKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	polynomialKRRCVParams.KernelCrossValidationTrainingParams.DegreeToTry = { 1.0 };
	auto const polynomialKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<PolynomialKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, polynomialKRRDiagnostics, polynomialKRRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(polynomialKRRRegressor), polynomialKRRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialKRRDiagnostics), polynomialKRRDiagnosticsMD5);

	std::vector<T> radialBasisKRRDiagnostics;
	RadialBasisKRR::CrossValidationTrainingParams radialBasisKRRCVParams;
	radialBasisKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	radialBasisKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	radialBasisKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	auto const radialBasisKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<RadialBasisKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, radialBasisKRRDiagnostics, radialBasisKRRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(radialBasisKRRRegressor), radialBasisKRRRegressorMD5);
	EXPECT_EQ(GetMD5(radialBasisKRRDiagnostics), radialBasisKRRDiagnosticsMD5);

	std::vector<T> sigmoidKRRDiagnostics;
	SigmoidKRR::CrossValidationTrainingParams sigmoidKRRCVParams;
	sigmoidKRRCVParams.MaxBasisFunctionsToTry = { 200, 400 };
	sigmoidKRRCVParams.LambdaToTry = { 1.e-6, 10.0 };
	sigmoidKRRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	sigmoidKRRCVParams.KernelCrossValidationTrainingParams.CoeffToTry = { 0.0, 1.0 };
	auto const sigmoidKRRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<SigmoidKRR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, sigmoidKRRDiagnostics, sigmoidKRRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(sigmoidKRRRegressor), sigmoidKRRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidKRRDiagnostics), sigmoidKRRDiagnosticsMD5);

	std::vector<T> linearSVRDiagnostics;
	LinearSVR::CrossValidationTrainingParams linearSVRCVParams;
	linearSVRCVParams.CToTry = { 1.0, 2.0 };
	linearSVRCVParams.EpsilonToTry = { 1.e-3 };
	linearSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	linearSVRCVParams.CacheSizeToTry = { 200 };
	auto const linearSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearSVRDiagnostics, linearSVRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
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
	auto const polynomialSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<PolynomialSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, polynomialSVRDiagnostics, polynomialSVRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(polynomialSVRRegressor), polynomialSVRRegressorMD5);
	EXPECT_EQ(GetMD5(polynomialSVRDiagnostics), polynomialSVRDiagnosticsMD5);

	std::vector<T> radialBasisSVRDiagnostics;
	RadialBasisSVR::CrossValidationTrainingParams radialBasisSVRCVParams;
	radialBasisSVRCVParams.CToTry = { 1.0, 2.0 };
	radialBasisSVRCVParams.EpsilonToTry = { 1.e-3 };
	radialBasisSVRCVParams.EpsilonInsensitivityToTry = { 0.1, 0.2 };
	radialBasisSVRCVParams.CacheSizeToTry = { 200 };
	radialBasisSVRCVParams.KernelCrossValidationTrainingParams.GammaToTry = { 1.0, 2.0 };
	auto const radialBasisSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<RadialBasisSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, radialBasisSVRDiagnostics, radialBasisSVRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
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
	auto const sigmoidSVRRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<SigmoidSVR>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, sigmoidSVRDiagnostics, sigmoidSVRCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(sigmoidSVRRegressor), sigmoidSVRRegressorMD5);
	EXPECT_EQ(GetMD5(sigmoidSVRDiagnostics), sigmoidSVRDiagnosticsMD5);

	std::vector<T> denseRFDiagnostics;
	DenseRF::CrossValidationTrainingParams denseRFCVParams;
	denseRFCVParams.NumTreesToTry = { 100, 1000 };
	denseRFCVParams.MinSamplesPerLeafToTry = { 3, 5 };
	denseRFCVParams.SubsamplingFractionToTry = { 1.0 / 3.0, 0.5 };
	auto const denseRFRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<DenseRF>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, denseRFDiagnostics, denseRFCVParams, normaliserCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(denseRFRegressor), denseRFRegressorMD5);
	EXPECT_EQ(GetMD5(denseRFDiagnostics), denseRFDiagnosticsMD5);

	std::vector<T> linearLogitIRLSDiagnostics;
	LinearLogitIRLS::CrossValidationTrainingParams linearLogitIRLSCVParams;
	linearLogitIRLSCVParams.LambdaToTry = { 1.e-3, 0.1 };
	linearLogitIRLSCVParams.MaxBasisFunctionsToTry = { 50, 100 };
	linearLogitIRLSCVParams.MaxNumIterationsToTry = { 100 };
	linearLogitIRLSCVParams.ConvergenceToleranceToTry = { 1e-4 };
	auto const linearLogitIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearLogitIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearLogitIRLSDiagnostics, linearLogitIRLSCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(linearLogitIRLSRegressor), linearLogitIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearLogitIRLSDiagnostics), linearLogitIRLSDiagnosticsMD5);

	std::vector<T> linearFourierIRLSDiagnostics;
	LinearFourierIRLS::CrossValidationTrainingParams linearFourierIRLSCVParams;
	linearFourierIRLSCVParams.LambdaToTry = { 1.e-3, 0.1 };
	linearFourierIRLSCVParams.MaxBasisFunctionsToTry = { 50, 100 };
	linearFourierIRLSCVParams.MaxNumIterationsToTry = { 100 };
	linearFourierIRLSCVParams.ConvergenceToleranceToTry = { 1e-4 };
	auto const linearFourierIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearFourierIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearFourierIRLSDiagnostics, linearFourierIRLSCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(linearFourierIRLSRegressor), linearFourierIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearFourierIRLSDiagnostics), linearFourierIRLSDiagnosticsMD5);

	std::vector<T> linearLagrangeIRLSDiagnostics;
	LinearLagrangeIRLS::CrossValidationTrainingParams linearLagrangeIRLSCVParams;
	linearLagrangeIRLSCVParams.LambdaToTry = { 1.e-3, 0.1 };
	linearLagrangeIRLSCVParams.MaxBasisFunctionsToTry = { 50, 100 };
	linearLagrangeIRLSCVParams.MaxNumIterationsToTry = { 100 };
	linearLagrangeIRLSCVParams.ConvergenceToleranceToTry = { 1e-4 };
	auto const linearLagrangeIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorCrossValidation<LinearLagrangeIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, numThreads, linearLagrangeIRLSDiagnostics, linearLagrangeIRLSCVParams, featureSelectionCVParams, PCACVParams);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSRegressor), linearLagrangeIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSDiagnostics), linearLagrangeIRLSDiagnosticsMD5);
}