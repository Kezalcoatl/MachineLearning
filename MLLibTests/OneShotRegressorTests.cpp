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
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::LogitLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearLogitIRLS;
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::FourierLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearFourierIRLS;
	typedef RegressionTypes::IterativelyReweightedLeastSquaresRegression<LinkFunctionTypes::LagrangeLinkFunction<KernelTypes::LinearKernel<SampleType>>> LinearLagrangeIRLS;

	static size_t const numExamples = 50;
	static size_t const numOrdinates = 10;

	std::vector<SampleType> inputExamples(numExamples, SampleType(numOrdinates));
	std::vector<T> targetExamples(numExamples);
	std::vector<SampleType> binaryClassificationInputExamples(numExamples, SampleType(1));
	std::vector<T> binaryClassificationTargetExamples(numExamples);

	std::string const randomSeed = "MLLib";
	dlib::rand RNG(randomSeed);

	for (size_t e = 0; e < numExamples; ++e)
	{
		for (size_t o = 0; o < numOrdinates; ++o)
		{
			T sinArg = static_cast<T>(e + 1);
			inputExamples[e](o) = std::sin(sinArg * sinArg) * std::exp(static_cast<T>(o));
		}
		targetExamples[e] = static_cast<T>(e + 1) + std::sin(static_cast<T>(e) * 0.1 * dlib::pi * 2.0);
		binaryClassificationInputExamples[e](0) = std::sin(static_cast<T>(e));
		binaryClassificationTargetExamples[e] = /*RNG.get_double_in_range(0.0, 0.1) +*/ binaryClassificationInputExamples[e](0) > 0 ? 1.0 : 0.0;
	}

	ECrossValidationMetric const metric = ECrossValidationMetric::SumSquareMean;
	size_t const numFolds = 4;
	size_t const maxNumCalls = 1000;
	size_t const numThreads = 16;
	std::string const linearKRRRegressorMD5 = "a8524cda078d2dfc1e97d0059b8f725e";
	std::string const linearKRRDiagnosticsMD5 = "1ff023f4a6c549cda0083397d0986831";
	std::string const polynomialKRRRegressorMD5 = "4c1b9431824cc62496d98c8d4ebe702a";
	std::string const polynomialKRRDiagnosticsMD5 = "1ff023f4a6c549cda0083397d0986831";
	std::string const radialBasisKRRRegressorMD5 = "67db812ba59c98796dc4c6ba7f144165";
	std::string const radialBasisKRRDiagnosticsMD5 = "a43429ddc4f1c871091f8d9bceff66f9";
	std::string const sigmoidKRRRegressorMD5 = "f524fc018acf3a98b9b9b08118e5f24a";
	std::string const sigmoidKRRDiagnosticsMD5 = "e4f98cb09b2300a8134b9df28fddbc13";
	std::string const linearSVRRegressorMD5 = "99d122c3add73b5177043da2459ea977";
	std::string const linearSVRDiagnosticsMD5 = "5754c786a1044105c0cbf1e4b430a72a";
	std::string const polynomialSVRRegressorMD5 = "e2ea42d340d2966d788b330c3a5102d9";
	std::string const polynomialSVRDiagnosticsMD5 = "5754c786a1044105c0cbf1e4b430a72a";
	std::string const radialBasisSVRRegressorMD5 = "1b6cbf0e6885bdd5df56f7b6af589d9e";
	std::string const radialBasisSVRDiagnosticsMD5 = "e4808470c2827efa985013ed42ac9f44";
	std::string const sigmoidSVRRegressorMD5 = "ed519c161597a73310b351b33c0f9b55";
	std::string const sigmoidSVRDiagnosticsMD5 = "e1d57e342b441896cf4d6af5a04ab582";
	std::string const denseRFRegressorMD5 = "b11ab694fe00a83557798be840533d2c";
	std::string const denseRFDiagnosticsMD5 = "6b14f26c45926af5e1f92821e4c12690";
	std::string const linearLogitIRLSRegressorMD5 = "";
	std::string const linearLogitIRLSDiagnosticsMD5 = "";
	std::string const linearFourierIRLSRegressorMD5 = "";
	std::string const linearFourierIRLSDiagnosticsMD5 = "";
	std::string const linearLagrangeIRLSRegressorMD5 = "";
	std::string const linearLagrangeIRLSDiagnosticsMD5 = "";

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

	std::vector<T> linearLogitIRLSDiagnostics;
	LinearLogitIRLS::OneShotTrainingParams linearLogitIRLSOSParams;
	linearLogitIRLSOSParams.MaxNumIterations = 100;
	linearLogitIRLSOSParams.ConvergenceTolerance = 1.e-4;
	linearLogitIRLSOSParams.MaxBasisFunctions = 100;
	linearLogitIRLSOSParams.Lambda = 1.e-3;
	auto const linearLogitIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearLogitIRLS>(binaryClassificationInputExamples, binaryClassificationTargetExamples, randomSeed, metric, numFolds, linearLogitIRLSDiagnostics, linearLogitIRLSOSParams);
	EXPECT_EQ(GetMD5(linearLogitIRLSRegressor), linearLogitIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearLogitIRLSDiagnostics), linearLogitIRLSDiagnosticsMD5);
	
	std::vector<T> linearFourierIRLSDiagnostics;
	LinearFourierIRLS::OneShotTrainingParams linearFourierIRLSOSParams;
	linearFourierIRLSOSParams.MaxNumIterations = 100;
	linearFourierIRLSOSParams.ConvergenceTolerance = 1.e-4;
	linearFourierIRLSOSParams.MaxBasisFunctions = 100;
	linearFourierIRLSOSParams.Lambda = 1.e-3;
	linearFourierIRLSOSParams.LinkFunctionOneShotTrainingParams.NumTerms = 38;
	std::vector<SampleType> fourierExamples(50, SampleType(1));
	for (size_t i = 0; i < fourierExamples.size(); ++i)
	{
		fourierExamples[i](0) = static_cast<T>(i);
	}
	auto const linearFourierIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearFourierIRLS>(fourierExamples, targetExamples, randomSeed, metric, numFolds, linearFourierIRLSDiagnostics, linearFourierIRLSOSParams);
	EXPECT_EQ(GetMD5(linearFourierIRLSRegressor), linearFourierIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearFourierIRLSDiagnostics), linearFourierIRLSDiagnosticsMD5);

	std::vector<T> linearLagrangeIRLSDiagnostics;
	LinearLagrangeIRLS::OneShotTrainingParams linearLagrangeIRLSOSParams;
	linearLagrangeIRLSOSParams.MaxNumIterations = 100;
	linearLagrangeIRLSOSParams.ConvergenceTolerance = 1.e-4;
	linearLagrangeIRLSOSParams.MaxBasisFunctions = 100;
	linearLagrangeIRLSOSParams.Lambda = 1.e-3;
	auto const linearLagrangeIRLSRegressor = Regressors::RegressorTrainer::TrainRegressorOneShot<LinearLagrangeIRLS>(inputExamples, targetExamples, randomSeed, metric, numFolds, linearLagrangeIRLSDiagnostics, linearLagrangeIRLSOSParams);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSRegressor), linearLagrangeIRLSRegressorMD5);
	EXPECT_EQ(GetMD5(linearLagrangeIRLSDiagnostics), linearLagrangeIRLSDiagnosticsMD5);
}