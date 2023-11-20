#pragma once

namespace Regressors
{
	namespace ModifierTypes
	{
		template <typename T>
		NormaliserModifier<T>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
		}

		template <typename T>
		NormaliserModifier<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename T>
		NormaliserModifier<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename T> template <size_t TotalNumParams>
		void NormaliserModifier<T>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset,
			FindMinGlobalTrainingParams const& fmgTrainingParams)
		{
		}

		template <typename T> template <size_t TotalNumParams>
		void NormaliserModifier<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
		{
		}

		template <typename T>
		unsigned NormaliserModifier<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvParams)
		{
			return 1u;
		}

		template <typename T>
		NormaliserModifier<T>::ModifierFunction::ModifierFunction(OneShotTrainingParams const& osParams,
			std::vector<col_vector<T>> const& inputExamples,
			std::vector<T> const& targetExamples)
		{
			Normaliser.train(inputExamples);
			TrainedParams = osParams;
		}

		template <typename T>
		void NormaliserModifier<T>::ModifierFunction::Modify(col_vector<T>& input) const
		{
			input = Normaliser(input);
		}

		template <typename T>
		typename NormaliserModifier<T>::OneShotTrainingParams const& NormaliserModifier<T>::ModifierFunction::GetTrainedParams() const
		{
			return TrainedParams;
		}

		/////////////////////////////////////////////////////////////////////////

		template <typename T>
		InputPCAModifier<T>::OneShotTrainingParams::OneShotTrainingParams() : TargetVariance(0.95)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		InputPCAModifier<T>::ModifierFunction::ModifierFunction(OneShotTrainingParams const& osTrainingParams, std::vector<col_vector<T>> const& inputExamples, std::vector<T> const& targetExamples)
		{
			PCAModel = PCA::PrincipalComponentAnalysisTrainer<col_vector<T>>::TrainToTargetVariance(inputExamples, osTrainingParams.TargetVariance, inputExamples.begin()->size());
			TrainedModifierParams = osTrainingParams;
		}

		template <typename T>
		InputPCAModifier<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			TargetVarianceToTry = { temp.TargetVariance };
		}

		template <typename T>
		InputPCAModifier<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerTargetVariance = temp.TargetVariance;
			UpperTargetVariance = temp.TargetVariance;
		}

		template <typename T> template <size_t TotalNumParams>
		void InputPCAModifier<T>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset,
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

		template <typename T> template <size_t TotalNumParams>
		void InputPCAModifier<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.TargetVariance = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.TargetVariance = optimiseParamsMap[mapOffset].second;
			}
		}

		template <typename T>
		unsigned InputPCAModifier<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.TargetVarianceToTry.size();
		}

		template <typename T>
		void InputPCAModifier<T>::ModifierFunction::Modify(col_vector<T>& input) const
		{
			input = PCAModel.Encode(input, PCAModel.nParams());
		}

		template <typename T>
		typename InputPCAModifier<T>::OneShotTrainingParams const& InputPCAModifier<T>::ModifierFunction::GetTrainedParams() const
		{
			return TrainedModifierParams;
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		FeatureSelectionModifier<T>::OneShotTrainingParams::OneShotTrainingParams() : FeatureFraction(0.5)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		std::vector<size_t> FeatureSelectionModifier<T>::ModifierFunction::GetOrderedCorrelationIndices(std::vector<col_vector<T>> const& inputExamples,
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

		template <typename T>
		FeatureSelectionModifier<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			FeatureFractionsToTry = { temp.FeatureFraction };
		}

		template <typename T>
		FeatureSelectionModifier<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerFeatureFraction = temp.FeatureFraction;
			UpperFeatureFraction = temp.FeatureFraction;
		}

		template <typename T>
		FeatureSelectionModifier<T>::ModifierFunction::ModifierFunction(OneShotTrainingParams const& osTrainingParams,
			std::vector<col_vector<T>> const& inputExamples,
			std::vector<T> const& targetExamples)
		{
			TrainedParams = osTrainingParams;
			FeatureIndices = GetOrderedCorrelationIndices(inputExamples, targetExamples, osTrainingParams.FeatureFraction);
		}

		template <typename T> template <size_t TotalNumParams>
		void FeatureSelectionModifier<T>::PackageParameters(col_vector<T>& lowerParams,
			col_vector<T>& upperParams,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset,
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

		template <typename T> template <size_t TotalNumParams>
		void FeatureSelectionModifier<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.FeatureFraction = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.FeatureFraction = optimiseParamsMap[mapOffset].second;
			}
		}

		template <typename T>
		unsigned FeatureSelectionModifier<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.FeatureFractionsToTry.size();
		}

		template <typename T>
		void FeatureSelectionModifier<T>::ModifierFunction::Modify(col_vector<T>& input) const
		{
			col_vector<T> featureSelectedInput(FeatureIndices.size());
			for (unsigned i = 0; i < FeatureIndices.size(); ++i)
			{
				featureSelectedInput(i) = input(FeatureIndices[i]);
			}
			input = featureSelectedInput;
		}

		template <typename T>
		typename FeatureSelectionModifier<T>::OneShotTrainingParams const& FeatureSelectionModifier<T>::ModifierFunction::GetTrainedParams() const
		{
			return TrainedParams;
		}
	}
}