#pragma once
#include <MLLib/TypeDefinitions.h>
#include <MLLib/PrincipalComponentAnalysis.h>
#include <dlib/statistics.h>

namespace Regressors
{
	enum class EModifierFunctionTypes;

	namespace ModifierTypes
	{
		template <class... ModifierCrossValidationTrainingTypes>
		static void IterateModifiers(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifiersCrossValidationTrainingParams,
			std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry);

		template <typename SampleType>
		class NormaliserModifier
		{
		public:
			typedef typename SampleType::type T;
			static EModifierFunctionTypes const ModifierTypeEnum;
			static size_t const NumModifierParams;

			struct OneShotTrainingParams : public RegressorTrainer::ModifierOneShotTrainingParamsBase
			{
				typedef NormaliserModifier ModifierType;

				OneShotTrainingParams();

				EModifierFunctionTypes GetModifierType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
				}
			};

			struct CrossValidationTrainingParams
			{
				typedef NormaliserModifier ModifierType;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				typedef NormaliserModifier ModifierType;

				FindMinGlobalTrainingParams();
			};

			struct ModifierFunction
			{
				typedef NormaliserModifier ModifierType;

				OneShotTrainingParams TrainingParams;

				dlib::vector_normalizer<SampleType> Normaliser;

				void operator()(SampleType& input) const;

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					serialize(item.TrainingParams, out);
					dlib::serialize(item.Normaliser, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					deserialize(item.TrainingParams, in);
					dlib::deserialize(item.Normaliser, in);
				}
			};

			static void TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples);

			template <size_t I, class... ModifierCrossValidationTrainingTypes>
			static void IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
				std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams);

			template <size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& params);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
			static void UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <typename SampleType>
		class InputPCAModifier
		{
		public:
			typedef typename SampleType::type T;
			static const EModifierFunctionTypes ModifierTypeEnum;
			static const size_t NumModifierParams;

			struct OneShotTrainingParams : public RegressorTrainer::ModifierOneShotTrainingParamsBase
			{
				typedef InputPCAModifier ModifierType;
				T TargetVariance;

				OneShotTrainingParams();

				EModifierFunctionTypes GetModifierType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.TargetVariance, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.TargetVariance, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				typedef InputPCAModifier ModifierType;
				std::vector<T> TargetVarianceToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				typedef InputPCAModifier ModifierType;
				T LowerTargetVariance;
				T UpperTargetVariance;

				FindMinGlobalTrainingParams();
			};

			struct ModifierFunction
			{
				typedef InputPCAModifier ModifierType;

				OneShotTrainingParams TrainingParams;

				PCA::PrincipalComponentAnalysis<SampleType> PCAModel;

				void operator()(SampleType& input) const;

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					serialize(item.TrainingParams, out);
					serialize(item.PCAModel, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					deserialize(item.TrainingParams, in);
					deserialize(item.PCAModel, in);
				}
			};

			static void TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples);

			template <size_t I, class... ModifierCrossValidationTrainingTypes>
			static void IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
				std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams);

			template <size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& params);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
			static void UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <typename SampleType>
		class FeatureSelectionModifier
		{
		public:
			typedef typename SampleType::type T;
			static const EModifierFunctionTypes ModifierTypeEnum;
			static const size_t NumModifierParams;

			struct OneShotTrainingParams : public RegressorTrainer::ModifierOneShotTrainingParamsBase
			{
				typedef FeatureSelectionModifier ModifierType;
				T FeatureFraction;

				OneShotTrainingParams();

				EModifierFunctionTypes GetModifierType() const override;

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.FeatureFraction, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.FeatureFraction, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				typedef FeatureSelectionModifier ModifierType;
				std::vector<T> FeatureFractionsToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				typedef FeatureSelectionModifier ModifierType;
				T LowerFeatureFraction;
				T UpperFeatureFraction;

				FindMinGlobalTrainingParams();
			};

			struct ModifierFunction
			{
				typedef FeatureSelectionModifier ModifierType;

				OneShotTrainingParams TrainingParams;

				std::vector<size_t> FeatureIndices;

				void operator()(SampleType& input) const;

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					serialize(item.TrainingParams, out);
					dlib::serialize(item.FeatureIndices, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					deserialize(item.TrainingParams, in);
					dlib::deserialize(item.FeatureIndices, in);
				}
			};

			static void TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples);

			static std::vector<size_t> GetOrderedCorrelationIndices(std::vector<SampleType> const& inputExamples,
				std::vector<T> const& targetExamples,
				T const& featureFraction);

			template <size_t I, class... ModifierCrossValidationTrainingTypes>
			static void IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
				std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams);

			template <size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& fmgParams);

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
			static void UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <typename SampleType>
		size_t const NormaliserModifier<SampleType>::NumModifierParams = 0ull;
		template <typename SampleType>
		EModifierFunctionTypes const NormaliserModifier<SampleType>::ModifierTypeEnum = EModifierFunctionTypes::normaliser;
		template <typename SampleType>
		size_t const InputPCAModifier<SampleType>::NumModifierParams = 1ull;
		template <typename SampleType>
		EModifierFunctionTypes const InputPCAModifier<SampleType>::ModifierTypeEnum = EModifierFunctionTypes::inputPCA;
		template <typename SampleType>
		size_t const FeatureSelectionModifier<SampleType>::NumModifierParams = 1ull;
		template <typename SampleType>
		EModifierFunctionTypes const FeatureSelectionModifier<SampleType>::ModifierTypeEnum = EModifierFunctionTypes::featureSelection;
	}
}

#include "impl/ModifierTypes.hpp"