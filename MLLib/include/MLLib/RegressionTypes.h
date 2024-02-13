#pragma once
#include <MLLib/TypeDefinitions.h>
#include <MLLib/KernelTypes.h>
#include <MLLib/GKMTrainer.h>
#include <dlib/svm.h>

namespace Regressors
{
	template <class RegressionType, class... ModifierFunctionTypes>
	class impl;

	class RegressorTrainer;

	enum class ERegressorTypes;

	namespace RegressionTypes
	{
		template <class KernelType>
		class KernelRidgeRegression
		{
		public:
			 ERegressorTypes static const RegressorTypeEnum;
			 typedef typename KernelType::SampleType SampleType;
			 typedef typename KernelType::SampleType::type T;
			 typedef dlib::decision_function<typename KernelType::KernelFunctionType> DecisionFunction;

			 KernelRidgeRegression() = delete;
			 size_t static const NumTotalParams;
			 size_t static const NumRegressionParams;

			 struct OneShotTrainingParams : public RegressorTrainer::RegressionOneShotTrainingParamsBase
			 {
				 unsigned long MaxBasisFunctions;
				 T Lambda;
				 typename KernelType::OneShotTrainingParams KernelOneShotTrainingParams;

				 OneShotTrainingParams();

				 template <size_t TotalNumParams>
				 OneShotTrainingParams(col_vector<T> const& vecParams,
					 std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					 size_t& paramsOffset);

				 ERegressorTypes GetRegressionType() const override;

				 friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				 {
					 dlib::serialize(item.MaxBasisFunctions, out);
					 dlib::serialize(item.Lambda, out);
					 serialize(item.KernelOneShotTrainingParams, out);
				 }

				 friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				 {
					 dlib::deserialize(item.MaxBasisFunctions, in);
					 dlib::deserialize(item.Lambda, in);
					 deserialize(item.KernelOneShotTrainingParams, in);
				 }
			 };

			 struct CrossValidationTrainingParams
			 {
				 std::vector<unsigned long> MaxBasisFunctionsToTry;
				 std::vector<T> LambdaToTry;
				 typename KernelType::CrossValidationTrainingParams KernelCrossValidationTrainingParams;

				 CrossValidationTrainingParams();
			 };

			 struct FindMinGlobalTrainingParams
			 {
				 unsigned long LowerMaxBasisFunctions;
				 unsigned long UpperMaxBasisFunctions;
				 T LowerLambda;
				 T UpperLambda;
				 typename KernelType::FindMinGlobalTrainingParams KernelFindMinGlobalTrainingParams;

				 FindMinGlobalTrainingParams();
			 };

			 template <class... ModifierFunctionTypes>
			 static impl<KernelRidgeRegression, ModifierFunctionTypes...> Train(std::vector<SampleType> const& inputExamples,
				 std::vector<T> const& targetExamples,
				 OneShotTrainingParams const& regressionTrainingParams,
				 std::vector<T>& LeaveOneOutValues,
				 T const& trainingError,
				 std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			 static void IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				 std::vector<OneShotTrainingParams>& regressionParamSets);

			 template <size_t TotalNumParams>
			 static void PackageParameters(col_vector<T>& lowerBound,
				 col_vector<T>& upperBound,
				 std::vector<bool>& isIntegerParam,
				 FindMinGlobalTrainingParams const& fmgTrainingParams,
				 std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				 size_t& paramsOffset);

			 template <size_t TotalNumParams>
			 static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				 std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class KernelType>
		class SupportVectorRegression
		{
		public:
			ERegressorTypes static const RegressorTypeEnum;
			typedef typename KernelType::SampleType SampleType;
			typedef typename KernelType::SampleType::type T;
			typedef dlib::decision_function<typename KernelType::KernelFunctionType> DecisionFunction;

			SupportVectorRegression() = delete;
			size_t static const NumTotalParams;
			size_t static const NumRegressionParams;

			struct OneShotTrainingParams : public RegressorTrainer::RegressionOneShotTrainingParamsBase
			{
				T C;
				T Epsilon;
				T EpsilonInsensitivity;
				long CacheSize;
				typename KernelType::OneShotTrainingParams KernelOneShotTrainingParams;

				OneShotTrainingParams();

				template <size_t TotalNumParams>
				OneShotTrainingParams(col_vector<T> const& vecParams,
					std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					size_t& paramsOffset);

				ERegressorTypes GetRegressionType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.C, out);
					dlib::serialize(item.Epsilon, out);
					dlib::serialize(item.EpsilonInsensitivity, out);
					dlib::serialize(item.CacheSize, out);
					serialize(item.KernelOneShotTrainingParams, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.C, in);
					dlib::deserialize(item.Epsilon, in);
					dlib::deserialize(item.EpsilonInsensitivity, in);
					dlib::deserialize(item.CacheSize, in);
					deserialize(item.KernelOneShotTrainingParams, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<T> CToTry;
				std::vector<T> EpsilonToTry;
				std::vector<T> EpsilonInsensitivityToTry;
				std::vector<long> CacheSizeToTry;
				typename KernelType::CrossValidationTrainingParams KernelCrossValidationTrainingParams;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				T LowerC;
				T UpperC;
				T LowerEpsilon;
				T UpperEpsilon;
				T LowerEpsilonInsensitivity;
				T UpperEpsilonInsensitivity;
				long LowerCacheSize;
				long UpperCacheSize;
				typename KernelType::FindMinGlobalTrainingParams KernelFindMinGlobalTrainingParams;

				FindMinGlobalTrainingParams();
			};

			template <class... ModifierFunctionTypes>
			static impl<SupportVectorRegression, ModifierFunctionTypes...> Train(std::vector<SampleType> const& inputExamples,
				std::vector<T> const& targetExamples,
				OneShotTrainingParams const& regressionTrainingParams,
				std::vector<T>& Residuals,
				T const& trainingError,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			static void IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				std::vector<OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class ExtractorType>
		class RandomForestRegression
		{
		public:
			ERegressorTypes static const RegressorTypeEnum;
			typedef typename ExtractorType::SampleType SampleType;
			typedef typename ExtractorType::SampleType::type T;
			typedef dlib::random_forest_regression_function<typename ExtractorType::ExtractorFunctionType> DecisionFunction;

			RandomForestRegression() = delete;
			size_t static const NumTotalParams;
			size_t static const NumRegressionParams;

			struct OneShotTrainingParams : public RegressorTrainer::RegressionOneShotTrainingParamsBase
			{
				size_t NumTrees;
				size_t MinSamplesPerLeaf;
				T SubsamplingFraction;
				typename ExtractorType::OneShotTrainingParams ExtractorOneShotTrainingParams;

				OneShotTrainingParams();

				template <size_t TotalNumParams>
				OneShotTrainingParams(col_vector<T> const& vecParams,
					std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					size_t& paramsOffset);

				ERegressorTypes GetRegressionType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.NumTrees, out);
					dlib::serialize(item.MinSamplesPerLeaf, out);
					dlib::serialize(item.SubsamplingFraction, out);
					serialize(item.ExtractorOneShotTrainingParams, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.NumTrees, in);
					dlib::deserialize(item.MinSamplesPerLeaf, in);
					dlib::deserialize(item.SubsamplingFraction, in);
					deserialize(item.ExtractorOneShotTrainingParams, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<size_t> NumTreesToTry;
				std::vector<size_t> MinSamplesPerLeafToTry;
				std::vector<T> SubsamplingFractionToTry;
				typename ExtractorType::CrossValidationTrainingParams ExtractorCrossValidationTrainingParams;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				size_t LowerNumTrees;
				size_t UpperNumTrees;
				size_t LowerMinSamplesPerLeaf;
				size_t UpperMinSamplesPerLeaf;
				T LowerSubsamplingFraction;
				T UpperSubsamplingFraction;
				typename ExtractorType::FindMinGlobalTrainingParams ExtractorFindMinGlobalTrainingParams;

				FindMinGlobalTrainingParams();
			};

			template <class... ModifierFunctionTypes>
			static impl<RandomForestRegression, ModifierFunctionTypes...> Train(std::vector<SampleType> const& inputExamples,
				std::vector<T> const& targetExamples,
				OneShotTrainingParams const& regressionTrainingParams,
				std::vector<T>& OutOfBagValues,
				T const& trainingError,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			static void IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				std::vector<OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class LinkFunctionType>
		class IterativelyReweightedLeastSquaresRegression
		{
		public:
			ERegressorTypes static const RegressorTypeEnum;
			typedef LinkFunctionType LinkFunctionType;
			typedef typename LinkFunctionType::KernelType KernelType;
			typedef typename KernelType::SampleType SampleType;
			typedef typename KernelType::SampleType::type T;
			typedef GKMDecisionFunction<LinkFunctionType> DecisionFunction;

			IterativelyReweightedLeastSquaresRegression() = delete;
			size_t static const NumTotalParams;
			size_t static const NumRegressionParams;

			struct OneShotTrainingParams : public RegressorTrainer::RegressionOneShotTrainingParamsBase
			{
				size_t MaxNumIterations;
				T ConvergenceTolerance;
				unsigned long MaxBasisFunctions;
				T Lambda;

				typename LinkFunctionType::OneShotTrainingParams LinkFunctionOneShotTrainingParams;
				typename KernelType::OneShotTrainingParams KernelOneShotTrainingParams;

				OneShotTrainingParams();

				template <size_t TotalNumParams>
				OneShotTrainingParams(col_vector<T> const& vecParams,
					std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					size_t& paramsOffset);

				ERegressorTypes GetRegressionType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.MaxNumIterations, out);
					dlib::serialize(item.ConvergenceTolerance, out);
					dlib::serialize(item.MaxBasisFunctions, out);
					dlib::serialize(item.Lambda, out);
					serialize(item.LinkFunctionOneShotTrainingParams, out);
					serialize(item.KernelOneShotTrainingParams, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.MaxNumIterations, in);
					dlib::deserialize(item.ConvergenceTolerance, in);
					dlib::deserialize(item.MaxBasisFunctions, in);
					dlib::deserialize(item.Lambda, in);
					deserialize(item.LinkFunctionOneShotTrainingParams, in);
					deserialize(item.KernelOneShotTrainingParams, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<size_t> MaxNumIterationsToTry;
				std::vector<T> ConvergenceToleranceToTry;
				std::vector<unsigned long> MaxBasisFunctionsToTry;
				std::vector<T> LambdaToTry;
				typename LinkFunctionType::CrossValidationTrainingParams LinkFunctionCrossValidationTrainingParams;
				typename KernelType::CrossValidationTrainingParams KernelCrossValidationTrainingParams;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				size_t LowerMaxNumIterations;
				size_t UpperMaxNumIterations;
				T LowerConvergenceTolerance;
				T UpperConvergenceTolerance;
				unsigned long LowerMaxBasisFunctions;
				unsigned long UpperMaxBasisFunctions;
				T LowerLambda;
				T UpperLambda;
				typename LinkFunctionType::FindMinGlobalTrainingParams LinkFunctionFindMinGlobalTrainingParams;
				typename KernelType::FindMinGlobalTrainingParams KernelFindMinGlobalTrainingParams;

				FindMinGlobalTrainingParams();
			};

			template <class... ModifierFunctionTypes>
			static impl<IterativelyReweightedLeastSquaresRegression, ModifierFunctionTypes...> Train(std::vector<SampleType> const& inputExamples,
				std::vector<T> const& targetExamples,
				OneShotTrainingParams const& regressionTrainingParams,
				std::vector<T>& Residuals,
				T const& trainingError,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			static void IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				std::vector<OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		template <class KernelType>
		size_t const KernelRidgeRegression<KernelType>::NumRegressionParams = 2ull;
		template <class KernelType>
		size_t const KernelRidgeRegression<KernelType>::NumTotalParams = NumRegressionParams + KernelType::NumKernelParams;
		template <class KernelType>
		ERegressorTypes const KernelRidgeRegression<KernelType>::RegressorTypeEnum =
			std::is_same<KernelType, KernelTypes::LinearKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::LinearKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::PolynomialKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::PolynomialKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::RadialBasisKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::RadialBasisKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::SigmoidKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::SigmoidKernelRidgeRegression :
			ERegressorTypes::MAX_NUMBER_OF_ERegressorTypes;

		template <class KernelType>
		size_t const SupportVectorRegression<KernelType>::NumRegressionParams = 4ull;
		template <class KernelType>
		size_t const SupportVectorRegression<KernelType>::NumTotalParams = NumRegressionParams + KernelType::NumKernelParams;
		template <class KernelType>
		ERegressorTypes const SupportVectorRegression<KernelType>::RegressorTypeEnum =
			std::is_same<KernelType, KernelTypes::LinearKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::LinearSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::PolynomialKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::PolynomialSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::RadialBasisKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::RadialBasisSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::SigmoidKernel<typename KernelType::SampleType>>::value ? ERegressorTypes::SigmoidSupportVectorRegression :
			ERegressorTypes::MAX_NUMBER_OF_ERegressorTypes;

		template <class ExtractorType>
		size_t const RandomForestRegression<ExtractorType>::NumRegressionParams = 3ull;
		template <class ExtractorType>
		size_t const RandomForestRegression<ExtractorType>::NumTotalParams = NumRegressionParams + ExtractorType::NumExtractorParams;
		template <class ExtractorType>
		ERegressorTypes const RandomForestRegression<ExtractorType>::RegressorTypeEnum =
			std::is_same<ExtractorType, KernelTypes::DenseExtractor<typename ExtractorType::SampleType>>::value ? ERegressorTypes::DenseRandomForestRegression :
			ERegressorTypes::MAX_NUMBER_OF_ERegressorTypes;

		template <class LinkFunctionType>
		size_t const IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::NumRegressionParams = 4ull;
		template <class LinkFunctionType>
		size_t const IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::NumTotalParams = NumRegressionParams + LinkFunctionType::NumLinkFunctionParams;
		template <class LinkFunctionType>
		ERegressorTypes const IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::RegressorTypeEnum = ERegressorTypes::MAX_NUMBER_OF_ERegressorTypes;
	}
}

#include "impl/RegressionTypes.hpp"