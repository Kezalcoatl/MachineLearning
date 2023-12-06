#pragma once

namespace Regressors
{
	namespace ModifierTypes
	{
		template <class... ModifierCrossValidationTrainingTypes>
		void ModifierComponentBase::IterateModifiers(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifiersCrossValidationTrainingParams,
			std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry)
		{
			if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) > 0)
			{
				std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...> modifierTrainingParams;
				using ModifierType = typename std::tuple_element<0, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
				ModifierType::IterateModifierParams<0>(modifiersCrossValidationTrainingParams, modifierOneShotTrainingParamsToTry, modifierTrainingParams);
			}
		}

		template <typename SampleType>
		NormaliserModifier<SampleType>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
		}

		template <typename SampleType>
		ModifierTypes NormaliserModifier<SampleType>::OneShotTrainingParams::GetModifierType() const
		{
			return NormaliserModifier::ModifierTypeEnum;
		}

		template <typename SampleType>
		NormaliserModifier<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename SampleType>
		NormaliserModifier<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename SampleType>
		void NormaliserModifier<SampleType>::TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples)
		{
			function.Normaliser.train(inputExamples);
			function.TrainingParams = params;
		}

		template <typename SampleType>
		void NormaliserModifier<SampleType>::ModifierFunction::operator()(SampleType& input) const
		{
			input = Normaliser(input);
		}

		template <typename SampleType> template <size_t I, class... ModifierCrossValidationTrainingTypes>
		void NormaliserModifier<SampleType>::IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
			std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
			std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams)
		{
			OneShotTrainingParams& iteratedParams = std::get<I>(modifierTrainingParams);
			if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) == I + 1)
			{
				modifierOneShotTrainingParamsToTry.emplace_back(modifierTrainingParams);
			}
			else
			{
				using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
				ModifierType::IterateModifierParams<I + 1>(modifierCrossValidationParams, modifierOneShotTrainingParamsToTry, modifierTrainingParams);
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		void NormaliserModifier<SampleType>::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& params)
		{

		}

		template <typename SampleType> template <size_t TotalNumParams>
		void NormaliserModifier<SampleType>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset,
			FindMinGlobalTrainingParams const& fmgTrainingParams)
		{
		}

		template <typename SampleType> template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
		void NormaliserModifier<SampleType>::UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			if constexpr (sizeof...(ModifierOneShotTrainingParams) == I + 1)
			{
				return;
			}
			else
			{
				using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierOneShotTrainingParams...>>::type::ModifierType;
				ModifierType::UnpackParameters<I + 1>(modifierOneShotTrainingParams,
					vecParams,
					optimiseParamsMap,
					mapOffset,
					paramsOffset);
			}
		}

		template <typename SampleType>
		size_t NormaliserModifier<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvParams)
		{
			return 1u;
		}

		/////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		InputPCAModifier<SampleType>::OneShotTrainingParams::OneShotTrainingParams() : TargetVariance(0.95)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		ModifierTypes InputPCAModifier<SampleType>::OneShotTrainingParams::GetModifierType() const
		{
			return InputPCAModifier::ModifierTypeEnum;
		}

		template <typename SampleType>
		InputPCAModifier<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			TargetVarianceToTry = { temp.TargetVariance };
		}

		template <typename SampleType>
		InputPCAModifier<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerTargetVariance = temp.TargetVariance;
			UpperTargetVariance = temp.TargetVariance;
		}

		template <typename SampleType>
		void InputPCAModifier<SampleType>::TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples)
		{
			function.PCAModel = PCA::PrincipalComponentAnalysisTrainer<col_vector<T>>::TrainToTargetVariance(inputExamples, params.TargetVariance, inputExamples.begin()->size());
			function.TrainingParams = params;
		}

		template <typename SampleType>
		void InputPCAModifier<SampleType>::ModifierFunction::operator()(SampleType& input) const
		{
			input = PCAModel.Encode(input, PCAModel.nParams());;
		}

		template <typename SampleType> template <size_t I, class... ModifierCrossValidationTrainingTypes>
		void InputPCAModifier<SampleType>::IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
			std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
			std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams)
		{
			OneShotTrainingParams& iteratedParams = std::get<I>(modifierTrainingParams);
			CrossValidationTrainingParams const& cvParams = std::get<I>(modifierCrossValidationParams);
			for (auto const& tv : cvParams.TargetVarianceToTry)
			{
				iteratedParams.TargetVariance = tv;
				if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) == I + 1)
				{
					modifierOneShotTrainingParamsToTry.emplace_back(modifierTrainingParams);
				}
				else
				{
					using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
					ModifierType::IterateModifierParams<I + 1>(modifierCrossValidationParams, modifierOneShotTrainingParamsToTry, modifierTrainingParams);
				}
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		void InputPCAModifier<SampleType>::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& params)
		{
			optimiseParamsMap[offset].first = params.LowerTargetVariance != params.UpperTargetVariance;
			optimiseParamsMap[offset].second = params.LowerTargetVariance;
		}

		template <typename SampleType> template <size_t TotalNumParams>
		void InputPCAModifier<SampleType>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset,
			FindMinGlobalTrainingParams const& fmgTrainingParams)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerParams(paramsOffset) = fmgTrainingParams.LowerTargetVariance;
				upperParams(paramsOffset) = fmgTrainingParams.UpperTargetVariance;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename SampleType> template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
		void InputPCAModifier<SampleType>::UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			auto& osTrainingParams = std::get<I>(modifierOneShotTrainingParams);
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.TargetVariance = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.TargetVariance = optimiseParamsMap[mapOffset].second;
			}
			if constexpr (sizeof...(ModifierOneShotTrainingParams) == I + 1)
			{
				return;
			}
			else
			{
				using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierOneShotTrainingParams...>>::type::ModifierType;
				ModifierType::UnpackParameters<I + 1>(modifierOneShotTrainingParams,
					vecParams,
					optimiseParamsMap,
					mapOffset,
					paramsOffset);
			}
		}

		template <typename SampleType>
		size_t InputPCAModifier<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.TargetVarianceToTry.size();
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		FeatureSelectionModifier<SampleType>::OneShotTrainingParams::OneShotTrainingParams() : FeatureFraction(0.5)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		ModifierTypes FeatureSelectionModifier<SampleType>::OneShotTrainingParams::GetModifierType() const
		{
			return FeatureSelectionModifier::ModifierTypeEnum;
		}

		template <typename SampleType>
		std::vector<size_t> FeatureSelectionModifier<SampleType>::GetOrderedCorrelationIndices(std::vector<SampleType> const& inputExamples,
			std::vector<T> const& targetExamples,
			T const& featureFraction)
		{
			size_t const numExamples = inputExamples.size();
			size_t const numOrdinates = inputExamples.begin()->nr();
			size_t const numFeatures = std::ceil(numOrdinates * featureFraction);
			std::vector<size_t> featureIndices(numOrdinates);
			std::iota(featureIndices.begin(), featureIndices.end(), 0);

			std::vector<T> univariateCoefficients(numOrdinates);
			for (size_t o = 0; o < numOrdinates; ++o)
			{
				T numeratorDot = 0.0;
				T denominatorDot = 0.0;
				for (size_t e = 0; e < numExamples; ++e)
				{
					numeratorDot += inputExamples[e](o) * targetExamples[e];
					denominatorDot += inputExamples[e](o) * inputExamples[e](o);
				}
				univariateCoefficients[o] = std::abs(numeratorDot / std::sqrt(denominatorDot));
			}
			std::sort(featureIndices.rbegin(), featureIndices.rend(), [&](size_t lhs, size_t rhs) -> bool { return univariateCoefficients[lhs] < univariateCoefficients[rhs]; });
			featureIndices.resize(numFeatures);
			return featureIndices;
		}

		template <typename SampleType>
		FeatureSelectionModifier<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			FeatureFractionsToTry = { temp.FeatureFraction };
		}

		template <typename SampleType>
		FeatureSelectionModifier<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerFeatureFraction = temp.FeatureFraction;
			UpperFeatureFraction = temp.FeatureFraction;
		}

		template <typename SampleType>
		void FeatureSelectionModifier<SampleType>::TrainModifier(ModifierFunction& function, OneShotTrainingParams const& params, std::vector<SampleType> const& inputExamples, std::vector<T> const& targetExamples)
		{
			function.FeatureIndices = GetOrderedCorrelationIndices(inputExamples, targetExamples, params.FeatureFraction);
			function.TrainingParams = params;
		}

		template <typename SampleType>
		void FeatureSelectionModifier<SampleType>::ModifierFunction::operator()(SampleType& input) const
		{
			SampleType featureSelectedInput(FeatureIndices.size());
			for (unsigned i = 0; i < FeatureIndices.size(); ++i)
			{
				featureSelectedInput(i) = input(FeatureIndices[i]);
			}
			input = featureSelectedInput;
		}

		template <typename SampleType> template <size_t I, class... ModifierCrossValidationTrainingTypes>
		void FeatureSelectionModifier<SampleType>::IterateModifierParams(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams,
			std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>>& modifierOneShotTrainingParamsToTry,
			std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>& modifierTrainingParams)
		{
			OneShotTrainingParams& iteratedParams = std::get<I>(modifierTrainingParams);
			CrossValidationTrainingParams const& cvParams = std::get<I>(modifierCrossValidationParams);
			for (auto const& ff : cvParams.FeatureFractionsToTry)
			{
				iteratedParams.FeatureFraction = ff;
				if constexpr (sizeof...(ModifierCrossValidationTrainingTypes) == I + 1)
				{
					modifierOneShotTrainingParamsToTry.emplace_back(modifierTrainingParams);
				}
				else
				{
					using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
					ModifierType::IterateModifierParams<I + 1>(modifierCrossValidationParams, modifierOneShotTrainingParamsToTry, modifierTrainingParams);
				}
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		void FeatureSelectionModifier<SampleType>::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, size_t const offset, FindMinGlobalTrainingParams const& fmgParams)
		{
			optimiseParamsMap[offset].first = fmgParams.LowerFeatureFraction != fmgParams.UpperFeatureFraction;
			optimiseParamsMap[offset].second = fmgParams.LowerFeatureFraction;
		}

		template <typename SampleType> template <size_t TotalNumParams>
		void FeatureSelectionModifier<SampleType>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset,
			FindMinGlobalTrainingParams const& fmgTrainingParams)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerParams(paramsOffset) = fmgTrainingParams.LowerFeatureFraction;
				upperParams(paramsOffset) = fmgTrainingParams.UpperFeatureFraction;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename SampleType> template <size_t I, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
		void FeatureSelectionModifier<SampleType>::UnpackParameters(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			auto& osTrainingParams = std::get<I>(modifierOneShotTrainingParams);
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.FeatureFraction = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.FeatureFraction = optimiseParamsMap[mapOffset].second;
			}
			if constexpr (sizeof...(ModifierOneShotTrainingParams) == I + 1)
			{
				return;
			}
			else
			{
				using ModifierType = typename std::tuple_element<I + 1, std::tuple<ModifierOneShotTrainingParams...>>::type::ModifierType;
				ModifierType::UnpackParameters<I + 1>(modifierOneShotTrainingParams,
					vecParams,
					optimiseParamsMap,
					mapOffset,
					paramsOffset);
			}
		}

		template <typename SampleType>
		size_t FeatureSelectionModifier<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.FeatureFractionsToTry.size();
		}
	}
}