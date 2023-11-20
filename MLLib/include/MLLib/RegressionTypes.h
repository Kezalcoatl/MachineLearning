#pragma once
#include <MLLib/TypeDefinitions.h>
#include <MLLib/KernelTypes.h>
#include <dlib/svm.h>

namespace Regressors
{
	template <typename T>
	class Regressor;

	namespace RegressionTypes
	{
		DECLARE_ENUM(RegressorTypes,
			LinearKernelRidgeRegression, 
			PolynomialKernelRidgeRegression,
			RadialBasisKernelRidgeRegression,
			SigmoidKernelRidgeRegression,
			LinearSupportVectorRegression,
			PolynomialSupportVectorRegression,
			RadialBasisSupportVectorRegression,
			SigmoidSupportVectorRegression,
			DenseRandomForestRegression);

		class RegressionComponentBase
		{
		public:
			RegressionComponentBase() = delete;

			struct RegressionOneShotTrainingParamsBase
			{
			private:
				RegressorTypes regressionType;
			public:
				virtual RegressorTypes const& GetRegressionType() const { return regressionType; }
			protected:
				RegressionOneShotTrainingParamsBase(RegressorTypes const& regressionType_)
					: regressionType(regressionType_)
				{
				}
			};

			template <class RegressionType, size_t I = 0, class... ModifierOneShotTrainingParamsTypes, class... ModifierFunctionTypes>
			static Regressor<typename RegressionType::T> TrainModifiersAndRegressor(std::vector<col_vector<typename RegressionType::T>> const& inputExamples,
				std::vector<typename RegressionType::T> const& targetExamples,
				typename RegressionType::OneShotTrainingParams const& regressionParams,
				std::vector<typename RegressionType::T>& diagnostics,
				typename RegressionType::T const& trainingError,
				std::tuple<ModifierOneShotTrainingParamsTypes...> const& modifierOneShotParams,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

		protected:
			template <class RegressionType, class... ModifierFunctionTypes>
			static Regressor<typename RegressionType::T> MakeRegressor(typename RegressionType::DecisionFunction const& function,
				std::tuple<ModifierFunctionTypes...> const& modifiers,
				typename RegressionType::T const& trainingError,
				typename RegressionType::OneShotTrainingParams const& regressorTrainingParams);

			template <class RegressionType, class... ModifierOneShotTrainingTypes>
			static Regressor<typename RegressionType::T> Train(std::vector<col_vector<typename RegressionType::T>> const& inputExamples,
				std::vector<typename RegressionType::T> const& targetExamples,
				std::string const& randomSeed,
				ModifierOneShotTrainingTypes const&... modifierOneShotParams);
		};

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class KernelType>
		class KernelRidgeRegression : public RegressionComponentBase
		{
		public:
			 RegressorTypes static const RegressorTypeEnum;
			 typedef typename KernelType::T T;
			 typedef dlib::decision_function<typename KernelType::KernelFunctionType> DecisionFunction;

			 KernelRidgeRegression() = delete;
			 size_t static const NumTotalParams;
			 size_t static const NumRegressionParams;

			 struct OneShotTrainingParams : public RegressionComponentBase::RegressionOneShotTrainingParamsBase
			 {
				 int MaxBasisFunctions;
				 T Lambda;
				 typename KernelType::OneShotTrainingParams KernelOneShotTrainingParams;

				 OneShotTrainingParams();

				 template <size_t TotalNumParams>
				 OneShotTrainingParams(col_vector<T> const& vecParams,
					 std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					 unsigned& paramsOffset);

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
					 deserialize(item.KernelOneShotTraininParams, in);
				 }
			 };

			 struct CrossValidationTrainingParams
			 {
				 std::vector<int> MaxBasisFunctionsToTry;
				 std::vector<T> LambdaToTry;
				 typename KernelType::CrossValidationTrainingParams KernelCrossValidationTrainingParams;

				 CrossValidationTrainingParams();
			 };

			 struct FindMinGlobalTrainingParams
			 {
				 int LowerMaxBasisFunctions;
				 int UpperMaxBasisFunctions;
				 T LowerLambda;
				 T UpperLambda;
				 typename KernelType::FindMinGlobalTrainingParams KernelFindMinGlobalTrainingParams;

				 FindMinGlobalTrainingParams();
			 };

			 template <class... ModifierFunctionTypes>
			 static Regressor<T> Train(std::vector<col_vector<T>> const& inputExamples,
				 std::vector<T> const& targetExamples,
				 OneShotTrainingParams const& regressionTrainingParams,
				 std::vector<T>& LeaveOneOutValues,
				 T const& trainingError,
				 std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			 static unsigned IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				 std::vector<OneShotTrainingParams>& regressionParamSets);

			 template <size_t TotalNumParams>
			 static void PackageParameters(col_vector<T>& lowerBound,
				 col_vector<T>& upperBound,
				 std::vector<bool>& isIntegerParam,
				 FindMinGlobalTrainingParams const& fmgTrainingParams,
				 std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				 unsigned& paramsOffset);

			 template <size_t TotalNumParams>
			 static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				 std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class KernelType>
		class SupportVectorRegression : public RegressionComponentBase
		{
		public:
			RegressorTypes static const RegressorTypeEnum;
			typedef typename KernelType::T T;
			typedef dlib::decision_function<typename KernelType::KernelFunctionType> DecisionFunction;

			SupportVectorRegression() = delete;
			size_t static const NumTotalParams;
			size_t static const NumRegressionParams;

			struct OneShotTrainingParams : public RegressionComponentBase::RegressionOneShotTrainingParamsBase
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
					unsigned& paramsOffset);

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
			static Regressor<T> Train(std::vector<col_vector<T>> const& inputExamples,
				std::vector<T> const& targetExamples,
				OneShotTrainingParams const& regressionTrainingParams,
				std::vector<T>& Residuals,
				T const& trainingError,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			static unsigned IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				std::vector<OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class ExtractorType>
		class RandomForestRegression : public RegressionComponentBase
		{
		public:
			RegressorTypes static const RegressorTypeEnum;
			typedef typename ExtractorType::T T;
			typedef dlib::random_forest_regression_function<typename ExtractorType::ExtractorFunctionType> DecisionFunction;

			RandomForestRegression() = delete;
			size_t static const NumTotalParams;
			size_t static const NumRegressionParams;

			struct OneShotTrainingParams : public RegressionComponentBase::RegressionOneShotTrainingParamsBase
			{
				size_t NumTrees;
				size_t MinSamplesPerLeaf;
				T SubsamplingFraction;
				typename ExtractorType::OneShotTrainingParams ExtractorOneShotTrainingParams;

				OneShotTrainingParams();

				template <size_t TotalNumParams>
				OneShotTrainingParams(col_vector<T> const& vecParams,
					std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
					unsigned& paramsOffset);

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
			static Regressor<T> Train(std::vector<col_vector<T>> const& inputExamples,
				std::vector<T> const& targetExamples,
				OneShotTrainingParams const& regressionTrainingParams,
				std::vector<T>& OutOfBagValues,
				T const& trainingError,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			static unsigned IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
				std::vector<OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap);
		};

		template <class KernelType>
		size_t const KernelRidgeRegression<KernelType>::NumRegressionParams = 2ull;
		template <class KernelType>
		size_t const KernelRidgeRegression<KernelType>::NumTotalParams = NumRegressionParams + KernelType::NumKernelParams;
		template <class KernelType>
		RegressorTypes const KernelRidgeRegression<KernelType>::RegressorTypeEnum =
			std::is_same<KernelType, KernelTypes::LinearKernel<typename KernelType::T>>::value ? RegressorTypes::LinearKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::PolynomialKernel<typename KernelType::T>>::value ? RegressorTypes::PolynomialKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::RadialBasisKernel<typename KernelType::T>>::value ? RegressorTypes::RadialBasisKernelRidgeRegression :
			std::is_same<KernelType, KernelTypes::SigmoidKernel<typename KernelType::T>>::value ? RegressorTypes::SigmoidKernelRidgeRegression :
			RegressorTypes::MAX_NUMBER_OF_RegressorTypes;

		template <class KernelType>
		size_t const SupportVectorRegression<KernelType>::NumRegressionParams = 4ull;
		template <class KernelType>
		size_t const SupportVectorRegression<KernelType>::NumTotalParams = NumRegressionParams + KernelType::NumKernelParams;
		template <class KernelType>
		RegressorTypes const SupportVectorRegression<KernelType>::RegressorTypeEnum =
			std::is_same<KernelType, KernelTypes::LinearKernel<typename KernelType::T>>::value ? RegressorTypes::LinearSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::PolynomialKernel<typename KernelType::T>>::value ? RegressorTypes::PolynomialSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::RadialBasisKernel<typename KernelType::T>>::value ? RegressorTypes::RadialBasisSupportVectorRegression :
			std::is_same<KernelType, KernelTypes::SigmoidKernel<typename KernelType::T>>::value ? RegressorTypes::SigmoidSupportVectorRegression :
			RegressorTypes::MAX_NUMBER_OF_RegressorTypes;

		template <class ExtractorType>
		size_t const RandomForestRegression<ExtractorType>::NumRegressionParams = 3ull;
		template <class ExtractorType>
		size_t const RandomForestRegression<ExtractorType>::NumTotalParams = NumRegressionParams + ExtractorType::NumExtractorParams;
		template <class ExtractorType>
		RegressorTypes const RandomForestRegression<ExtractorType>::RegressorTypeEnum =
			std::is_same<ExtractorType, KernelTypes::DenseExtractor<typename ExtractorType::T>>::value ? RegressorTypes::DenseRandomForestRegression :
			RegressorTypes::MAX_NUMBER_OF_RegressorTypes;
	}
}

#include "impl/RegressionTypes.hpp"