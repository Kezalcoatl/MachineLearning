#pragma once
#include <MLLib/TypeDefinitions.h>
#include <MLLib/PrincipalComponentAnalysis.h>
#include <dlib/statistics.h>

namespace Regressors
{
	template <typename T>
	class Regressor;
	enum class CrossValidationMetric;
	class RegressorTrainer;

	namespace ModifierTypes
	{
		DECLARE_ENUM(ModifierTypes,
			normaliser,
			inputPCA,
			featureSelection);

		template <typename T>
		class NormaliserModifier;
		template <typename T>
		class FeatureSelectionModifier;
		template <typename T>
		class InputPCAModifier;

		class ModifierComponentBase
		{
		public:
			ModifierComponentBase() = delete;

			struct ModifierOneShotTrainingParamsBase
			{
			private:
				ModifierTypes ModifierTypeEnum;

			public:
				virtual ModifierTypes const& GetModifierType() const { return ModifierTypeEnum; }
			};
		};

		template <typename T>
		class NormaliserModifier
		{
		public:
			static ModifierTypes const ModifierTypeEnum;
			static size_t const NumModifierParams;

			struct OneShotTrainingParams : public ModifierComponentBase::ModifierOneShotTrainingParamsBase
			{
				typedef NormaliserModifier ModifierType;

				OneShotTrainingParams();

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

			class ModifierFunction
			{
			private:
				dlib::vector_normalizer<col_vector<T>> Normaliser;
				OneShotTrainingParams TrainedParams;
			public:
				ModifierFunction() = default;
				ModifierFunction(OneShotTrainingParams const& oneShotParams,
					std::vector<col_vector<T>> const& inputExamples,
					std::vector<T> const& targetExamples);

				void Modify(col_vector<T>& input) const;

				OneShotTrainingParams const& GetTrainedParams() const;

				static ModifierTypes GetModifierType()
				{
					return NormaliserModifier::ModifierTypeEnum;
				}

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					dlib::serialize(item.Normaliser, out);
					serialize(item.TrainedParams, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					dlib::deserialize(item.normaliser, in);
					deserialize(item.trainedParams, in);
				}
			};

			template <size_t I = 0, class func, class... ModifierCrossValidationTrainingTypes, class... ModifierOneShotTrainingTypes>
			static void IterateModifierParams(func callback,
				std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::tuple<ModifierOneShotTrainingTypes...> const& modifierOneShotParams)
			{
				OneShotTrainingParams iteratedParams;
				auto expandedTuple = std::tuple_cat(modifierOneShotParams, std::make_tuple(iteratedParams));
				if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) == I + 1)
				{
					callback(expandedTuple);
				}
				else
				{
					using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
					ModifierType::IterateModifierParams<I + 1>(callback, modifierCrossValidationParams, expandedTuple);
				}
			}

			template <typename T, size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, unsigned const offset, FindMinGlobalTrainingParams const& params)
			{

			}

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		template <typename T>
		class InputPCAModifier
		{
		public:
			static const ModifierTypes ModifierTypeEnum;
			static const size_t NumModifierParams;

			struct OneShotTrainingParams : public ModifierComponentBase::ModifierOneShotTrainingParamsBase
			{
				typedef InputPCAModifier ModifierType;
				T TargetVariance;

				OneShotTrainingParams();

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

			class ModifierFunction
			{
			private:
				PCA::PrincipalComponentAnalysis<col_vector<T>> PCAModel;
				OneShotTrainingParams TrainedModifierParams;
			public:
				ModifierFunction() = default;
				ModifierFunction(OneShotTrainingParams const& osTrainingParams, std::vector<col_vector<T>> const& inputExamples, std::vector<T> const& targetExamples);

				void Modify(col_vector<T>& input) const;

				OneShotTrainingParams const& GetTrainedParams() const;

				static ModifierTypes GetModifierType()
				{
					return InputPCAModifier::ModifierTypeEnum;
				}

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					serialize(item.PCAModel, out);
					serialize(item.TrainedModifierParams, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					deserialize(item.PCAModel, in);
					deserialize(item.TrainedModifierParams, in);
				}
			};

			template <size_t I = 0, class func, class... ModifierCrossValidationTrainingTypes, class... ModifierOneShotTrainingTypes>
			static void IterateModifierParams(func callback,
				std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::tuple<ModifierOneShotTrainingTypes...> const& modifierOneShotParams)
			{
				OneShotTrainingParams iteratedParams;
				for (const auto& tv : std::get<I>(modifierCrossValidationParams).TargetVarianceToTry)
				{
					iteratedParams.TargetVariance = tv;
					auto expandedTuple = std::tuple_cat(modifierOneShotParams, std::make_tuple(iteratedParams));
					if constexpr (sizeof... (ModifierCrossValidationTrainingTypes) == I + 1)
					{
						callback(expandedTuple);
					}
					else
					{
						using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
						ModifierType::IterateModifierParams<I + 1>(callback, modifierCrossValidationParams, expandedTuple);
					}
				}
			}

			template <typename T, size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, unsigned const offset, FindMinGlobalTrainingParams const& params)
			{
				optimiseParamsMap[offset].first = params.LowerTargetVariance != params.UpperTargetVariance;
				optimiseParamsMap[offset].second = params.LowerTargetVariance;
			}

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		template <typename T>
		class FeatureSelectionModifier
		{
		public:
			static const ModifierTypes ModifierTypeEnum;
			static const size_t NumModifierParams;

			struct OneShotTrainingParams : public ModifierComponentBase::ModifierOneShotTrainingParamsBase
			{
				friend class FeatureSelectionModifier;
				typedef FeatureSelectionModifier ModifierType;
				T FeatureFraction;

				OneShotTrainingParams();

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

			class ModifierFunction
			{
			private:
				std::vector<size_t> FeatureIndices;
				OneShotTrainingParams TrainedParams;

				static std::vector<size_t> GetOrderedCorrelationIndices(std::vector<col_vector<T>> const& inputExamples,
					std::vector<T> const& targetExamples,
					T const& featureFraction);

			public:
				ModifierFunction() = default;
				ModifierFunction(OneShotTrainingParams const& osParams, std::vector<col_vector<T>> const& inputExamples, std::vector<T> const& targetExamples);

				void Modify(col_vector<T>& input) const;

				OneShotTrainingParams const& GetTrainedParams() const;

				static ModifierTypes GetModifierType()
				{
					return FeatureSelectionModifier::ModifierTypeEnum;
				}

				friend void serialize(ModifierFunction const& item, std::ostream& out)
				{
					dlib::serialize(item.FeatureIndices, out);
					serialize(item.TrainedParams, out);
				}

				friend void deserialize(ModifierFunction& item, std::istream& in)
				{
					dlib::deserialize(item.FeatureIndices, in);
					deserialize(item.TrainedParams, in);
				}
			};

			template <size_t I = 0, class func, class... ModifierCrossValidationTrainingTypes, class... ModifierOneShotTrainingTypes>
			static void IterateModifierParams(func callback,
				std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
				std::tuple<ModifierOneShotTrainingTypes...> const& modifierOneShotParams)
			{
				OneShotTrainingParams iteratedParams;
				for (const auto& ff : std::get<I>(modifierCrossValidationParams).FeatureFractionsToTry)
				{
					iteratedParams.FeatureFraction = ff;
					auto expandedTuple = std::tuple_cat(modifierOneShotParams, std::make_tuple(iteratedParams));
					if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) == I + 1)
					{
						callback(expandedTuple);
					}
					else
					{
						using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
						ModifierType::IterateModifierParams<I + 1>(callback, modifierCrossValidationParams, expandedTuple);
					}
				}
			}

			template <typename T, size_t TotalNumParams>
			static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, unsigned const offset, FindMinGlobalTrainingParams const& fmgParams)
			{
				optimiseParamsMap[offset].first = fmgParams.LowerFeatureFraction != fmgParams.UpperFeatureFraction;
				optimiseParamsMap[offset].second = fmgParams.LowerFeatureFraction;
			}

			template <size_t TotalNumParams>
			static void PackageParameters(col_vector<T>& lowerParams,
				col_vector<T>& upperParams,
				std::vector<bool>& isIntegerParam,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset,
				FindMinGlobalTrainingParams const& fmgTrainingParams);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(const CrossValidationTrainingParams& cvParams);
		};

		template <typename T>
		size_t const NormaliserModifier<T>::NumModifierParams = 0ull;
		template <typename T>
		ModifierTypes const NormaliserModifier<T>::ModifierTypeEnum = ModifierTypes::normaliser;
		template <typename T>
		size_t const InputPCAModifier<T>::NumModifierParams = 1ull;
		template <typename T>
		ModifierTypes const InputPCAModifier<T>::ModifierTypeEnum = ModifierTypes::inputPCA;
		template <typename T>
		size_t const FeatureSelectionModifier<T>::NumModifierParams = 1ull;
		template <typename T>
		ModifierTypes const FeatureSelectionModifier<T>::ModifierTypeEnum = ModifierTypes::featureSelection;
	}
}

#include "impl/ModifierTypes.hpp"