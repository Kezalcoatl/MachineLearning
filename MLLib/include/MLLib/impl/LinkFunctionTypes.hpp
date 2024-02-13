#pragma once

namespace Regressors
{
	namespace LinkFunctionTypes
	{
		template <typename KernelType>
		LogitLinkFunction<KernelType>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<ScalarType>::value, "ScalarType must be a floating point type.");
		}

		template <typename KernelType>
		LogitLinkFunction<KernelType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename KernelType>
		LogitLinkFunction<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename KernelType>
		typename LogitLinkFunction<KernelType>::LinkFunction LogitLinkFunction<KernelType>::Train(OneShotTrainingParams const& osParams,
			col_vector<ScalarType> const& inputExamples,
			col_vector<ScalarType> const& targetExamples)
		{
			return LinkFunction();
		}

		template <typename KernelType>
		typename LogitLinkFunction<KernelType>::ScalarType LogitLinkFunction<KernelType>::LinkFunction::operator()(ScalarType const& input) const
		{
			return 1.0 / (1.0 + std::exp(-input));
		}

		template <typename KernelType>
		template <class RegressionType>
		static void LogitLinkFunction<KernelType>::IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
			typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			KernelType::template IterateKernelParams<RegressionType>(regressionOneShotTrainingParams, kernelCrossValidationTrainingParams, regressionParamSets);
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LogitLinkFunction<KernelType>::PackageParameters(size_t const mapOffset,
			col_vector<ScalarType>& lowerBound,
			col_vector<ScalarType>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LogitLinkFunction<KernelType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LogitLinkFunction<KernelType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<ScalarType> const& vecParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename KernelType>
		FourierLinkFunction<KernelType>::OneShotTrainingParams::OneShotTrainingParams() : NumTerms(10ull)
		{
			static_assert(std::is_floating_point<ScalarType>::value, "ScalarType must be a floating point type.");
		}

		template <typename KernelType>
		FourierLinkFunction<KernelType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			NumTermsToTry = { temp.NumTerms };
		}

		template <typename KernelType>
		FourierLinkFunction<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerNumTerms = temp.NumTerms;
			UpperNumTerms = temp.NumTerms;
		}

		template <typename KernelType>
		typename FourierLinkFunction<KernelType>::LinkFunction FourierLinkFunction<KernelType>::Train(OneShotTrainingParams const& osParams,
			col_vector<ScalarType> const& inputExamples,
			col_vector<ScalarType> const& targetExamples)
		{
			DLIB_ASSERT(inputExamples.size() == targetExamples.size(),
				"");
			LinkFunction link;
			link.Min = std::numeric_limits<ScalarType>::max();
			link.Max = -std::numeric_limits<ScalarType>::max();
			link.MinStep = std::numeric_limits<ScalarType>::max();
			size_t const numExamples = inputExamples.size();
			bool useFFT = true;
			auto examples = inputExamples;
			std::sort(examples.begin(), examples.end());
			link.Min = examples(0);
			link.Max = examples(numExamples - 1ull);
			for (size_t index = 1ull; index < numExamples; ++index)
			{
				auto const step = std::abs(examples(index) - examples(index - 1ull));
				if (link.MinStep != std::numeric_limits<ScalarType>::max())
				{
					useFFT &= std::abs(link.MinStep - step) < std::numeric_limits<ScalarType>::epsilon();
				}
				if (step < link.MinStep)
				{
					link.MinStep = step;
				}
			}
			size_t const numCoefficients = 2ull * osParams.NumTerms + 1ull;
			dlib::matrix<ScalarType> data(numExamples, numCoefficients);
			dlib::set_colm(data, 0) = 1.0;
			for (size_t row = 0ull; row < numExamples; ++row)
			{
				auto const remappedInput = link.Theta(inputExamples(row));
				for (size_t k = 1ull; k <= osParams.NumTerms; ++k)
				{
					data(row, 2ull * k - 1ull) = std::cos(k * remappedInput);
					data(row, 2ull * k) = std::sin(k * remappedInput);
				}
			}
			col_vector<ScalarType> const coeffs = dlib::pinv(dlib::trans(data) * data) * dlib::trans(data) * targetExamples;
			link.Bias = coeffs(0);
			link.TrainingParams = osParams;
			link.Coefficients.resize(osParams.NumTerms);
			for (size_t term = 0; term < osParams.NumTerms; ++term)
			{
				link.Coefficients[term].first = coeffs(2ull * term + 1ull);
				link.Coefficients[term].second = coeffs(2ull * term + 2ull);
			}
			return link;
		}

		template <typename KernelType>
		typename FourierLinkFunction<KernelType>::ScalarType FourierLinkFunction<KernelType>::LinkFunction::Theta(ScalarType const& input) const
		{
			return 2.0 * dlib::pi * (input - Min) / (Max - Min + MinStep);
		}

		template <typename KernelType>
		typename FourierLinkFunction<KernelType>::ScalarType FourierLinkFunction<KernelType>::LinkFunction::operator()(ScalarType const& input) const
		{
			auto ret = Bias;
			auto const theta = Theta(input);
			for (size_t term = 0; term < Coefficients.size(); ++term)
			{
				auto const remappedInput = static_cast<ScalarType>(term + 1) * theta;
				ret += Coefficients[term].first * std::cos(remappedInput) + Coefficients[term].second * std::sin(remappedInput);
			}
			return ret;
		}

		template <typename KernelType>
		template <class RegressionType>
		static void FourierLinkFunction<KernelType>::IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
			typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			for (const auto& nt : linkFunctionCrossValidationTrainingParams.NumTermsToTry)
			{
				regressionOneShotTrainingParams.LinkFunctionOneShotTrainingParams.NumTerms = nt;
				KernelType::template IterateKernelParams<RegressionType>(regressionOneShotTrainingParams, kernelCrossValidationTrainingParams, regressionParamSets);
			}
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void FourierLinkFunction<KernelType>::PackageParameters(size_t const mapOffset,
			col_vector<ScalarType>& lowerBound,
			col_vector<ScalarType>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerNumTerms;
				upperBound(paramsOffset) = fmgTrainingParams.UpperNumTerms;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void FourierLinkFunction<KernelType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerNumTerms != fmgTrainingParams.UpperNumTerms;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerNumTerms;
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void FourierLinkFunction<KernelType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<ScalarType> const& vecParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.NumTerms = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.NumTerms = optimiseParamsMap[mapOffset].second;
			}
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename KernelType>
		LagrangeLinkFunction<KernelType>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<ScalarType>::value, "ScalarType must be a floating point type.");
		}

		template <typename KernelType>
		LagrangeLinkFunction<KernelType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename KernelType>
		LagrangeLinkFunction<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename KernelType>
		typename LagrangeLinkFunction<KernelType>::LinkFunction LagrangeLinkFunction<KernelType>::Train(OneShotTrainingParams const& osParams,
			col_vector<ScalarType> const& inputExamples,
			col_vector<ScalarType> const& targetExamples)
		{
			// TODO check for duplicates in inputExamples
			size_t const n = inputExamples.size();
			LinkFunction link;
			link.Weights.assign(n, 1.0);
			link.Roots.resize(n);
			link.Anchors.resize(n);
			for (size_t polyIndex = 0ull; polyIndex < n; ++polyIndex)
			{
				auto& weight = link.Weights[polyIndex];
				link.Roots[polyIndex] = inputExamples(polyIndex);
				link.Anchors[polyIndex] = targetExamples(polyIndex);
				for (size_t linearFactor = 0ull; linearFactor < n; ++linearFactor)
				{
					if (linearFactor == polyIndex) continue;
					weight /= inputExamples(polyIndex) - inputExamples(linearFactor);
				}
			}
			return link;
		}

		template <typename KernelType>
		typename LagrangeLinkFunction<KernelType>::ScalarType LagrangeLinkFunction<KernelType>::LinkFunction::operator()(ScalarType const& input) const
		{
			ScalarType numerator = 0.0;
			ScalarType denominator = 0.0;
			for (size_t poly = 0ull; poly < Weights.size(); ++poly)
			{
				ScalarType const term = Weights[poly] / (input - Roots[poly]);
				numerator += term * Anchors[poly];
				denominator += term;
				if (input == Roots[poly]) return Anchors[poly];
			}
			return numerator / denominator;
		}

		template <typename KernelType>
		template <class RegressionType>
		static void LagrangeLinkFunction<KernelType>::IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
			typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			KernelType::template IterateKernelParams<RegressionType>(regressionOneShotTrainingParams, kernelCrossValidationTrainingParams, regressionParamSets);
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LagrangeLinkFunction<KernelType>::PackageParameters(size_t const mapOffset,
			col_vector<ScalarType>& lowerBound,
			col_vector<ScalarType>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LagrangeLinkFunction<KernelType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
		}

		template <typename KernelType>
		template <size_t TotalNumParams>
		static void LagrangeLinkFunction<KernelType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<ScalarType> const& vecParams,
			std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
}