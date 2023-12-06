#pragma once

namespace Regressors
{
	namespace KernelTypes
	{
		template <typename SampleType>
		LinearKernel<SampleType>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
		}

		template <typename SampleType>
		LinearKernel<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename SampleType>
		LinearKernel<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename SampleType>
		typename LinearKernel<SampleType>::KernelFunctionType LinearKernel<SampleType>::GetKernel(OneShotTrainingParams const& osParams)
		{
			return KernelFunctionType();
		}

		template <typename SampleType> template <class RegressionType>
		static void LinearKernel<SampleType>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			regressionParamSets.emplace_back(regressionOneShotTrainingParams);
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void LinearKernel<SampleType>::PackageParameters(size_t const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void LinearKernel<SampleType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void LinearKernel<SampleType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
		}

		template <typename SampleType>
		size_t LinearKernel<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return 1u;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		PolynomialKernel<SampleType>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0), Coeff(0.0), Degree(1.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		PolynomialKernel<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
			CoeffToTry = { temp.Coeff };
			DegreeToTry = { temp.Degree };
		}

		template <typename SampleType>
		PolynomialKernel<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
			LowerCoeff = temp.Coeff;
			UpperCoeff = temp.Coeff;
			LowerDegree = temp.Degree;
			UpperDegree = temp.Degree;
		}

		template <typename SampleType>
		typename PolynomialKernel<SampleType>::KernelFunctionType PolynomialKernel<SampleType>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma, osTrainingParams.Coeff, osTrainingParams.Degree);
		}

		template <typename SampleType> template <class RegressionType>
		static void PolynomialKernel<SampleType>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(kernelCrossValidationTrainingParams.GammaToTry.size() > 0 && kernelCrossValidationTrainingParams.CoeffToTry.size() > 0 && kernelCrossValidationTrainingParams.DegreeToTry.size() > 0,
				"Every kernel parameter must have at least one value to try for cross-validation");
			for (const auto& g : kernelCrossValidationTrainingParams.GammaToTry)
			{
				regressionOneShotTrainingParams.KernelOneShotTrainingParams.Gamma = g;
				for (const auto& c : kernelCrossValidationTrainingParams.CoeffToTry)
				{
					regressionOneShotTrainingParams.KernelOneShotTrainingParams.Coeff = c;
					for (const auto& d : kernelCrossValidationTrainingParams.DegreeToTry)
					{
						regressionOneShotTrainingParams.KernelOneShotTrainingParams.Degree = d;
						regressionParamSets.emplace_back(regressionOneShotTrainingParams);
					}
				}
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void PolynomialKernel<SampleType>::PackageParameters(size_t const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerGamma;
				upperBound(paramsOffset) = fmgTrainingParams.UpperGamma;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[mapOffset + 1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerCoeff;
				upperBound(paramsOffset) = fmgTrainingParams.UpperCoeff;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[mapOffset + 2].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerDegree;
				upperBound(paramsOffset) = fmgTrainingParams.UpperDegree;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void PolynomialKernel<SampleType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
			optimiseParamsMap[mapOffset + 1].first = fmgTrainingParams.LowerCoeff != fmgTrainingParams.UpperCoeff;
			optimiseParamsMap[mapOffset + 1].second = fmgTrainingParams.LowerCoeff;
			optimiseParamsMap[mapOffset + 2].first = fmgTrainingParams.LowerDegree != fmgTrainingParams.UpperDegree;
			optimiseParamsMap[mapOffset + 2].second = fmgTrainingParams.LowerDegree;
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void PolynomialKernel<SampleType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.Gamma = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Gamma = optimiseParamsMap[mapOffset].second;
			}
			if (optimiseParamsMap[mapOffset + 1].first)
			{
				osTrainingParams.Coeff = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Coeff = optimiseParamsMap[mapOffset + 1].second;
			}
			if (optimiseParamsMap[mapOffset + 2].first)
			{
				osTrainingParams.Degree = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Degree = optimiseParamsMap[mapOffset + 2].second;
			}
		}

		template <typename SampleType>
		size_t PolynomialKernel<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size() * cvTrainingParams.CoeffToTry.size() * cvTrainingParams.DegreeToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		RadialBasisKernel<SampleType>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		RadialBasisKernel<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
		}

		template <typename SampleType>
		RadialBasisKernel<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
		}

		template <typename SampleType>
		typename RadialBasisKernel<SampleType>::KernelFunctionType RadialBasisKernel<SampleType>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma);
		}

		template <typename SampleType> template <class RegressionType>
		static void RadialBasisKernel<SampleType>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(kernelCrossValidationTrainingParams.GammaToTry.size() > 0,
				"Every kernel parameter must have at least one value to try for cross-validation");
			for (const auto& g : kernelCrossValidationTrainingParams.GammaToTry)
			{
				regressionOneShotTrainingParams.KernelOneShotTrainingParams.Gamma = g;
				regressionParamSets.emplace_back(regressionOneShotTrainingParams);
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void RadialBasisKernel<SampleType>::PackageParameters(size_t const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerGamma;
				upperBound(paramsOffset) = fmgTrainingParams.UpperGamma;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void RadialBasisKernel<SampleType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void RadialBasisKernel<SampleType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.Gamma = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Gamma = optimiseParamsMap[mapOffset].second;
			}
		}

		template <typename SampleType>
		size_t RadialBasisKernel<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		SigmoidKernel<SampleType>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0), Coeff(0.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		SigmoidKernel<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
			CoeffToTry = { temp.Coeff };
		}

		template <typename SampleType>
		SigmoidKernel<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
			LowerCoeff = temp.Coeff;
			UpperCoeff = temp.Coeff;
		}

		template <typename SampleType>
		typename SigmoidKernel<SampleType>::KernelFunctionType SigmoidKernel<SampleType>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma, osTrainingParams.Coeff);
		}

		template <typename SampleType> template <class RegressionType>
		static void SigmoidKernel<SampleType>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(kernelCrossValidationTrainingParams.GammaToTry.size() > 0 && kernelCrossValidationTrainingParams.CoeffToTry.size() > 0,
				"Every kernel parameter must have at least one value to try for cross-validation");
			for (const auto& g : kernelCrossValidationTrainingParams.GammaToTry)
			{
				regressionOneShotTrainingParams.KernelOneShotTrainingParams.Gamma = g;
				for (const auto& c : kernelCrossValidationTrainingParams.CoeffToTry)
				{
					regressionOneShotTrainingParams.KernelOneShotTrainingParams.Coeff = c;
					regressionParamSets.emplace_back(regressionOneShotTrainingParams);
				}
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void SigmoidKernel<SampleType>::PackageParameters(size_t const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerGamma;
				upperBound(paramsOffset) = fmgTrainingParams.UpperGamma;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[mapOffset + 1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerCoeff;
				upperBound(paramsOffset) = fmgTrainingParams.UpperCoeff;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void SigmoidKernel<SampleType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
			optimiseParamsMap[mapOffset + 1].first = fmgTrainingParams.LowerCoeff != fmgTrainingParams.UpperCoeff;
			optimiseParamsMap[mapOffset + 1].second = fmgTrainingParams.LowerCoeff;
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void SigmoidKernel<SampleType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				osTrainingParams.Gamma = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Gamma = optimiseParamsMap[mapOffset].second;
			}
			if (optimiseParamsMap[mapOffset + 1].first)
			{
				osTrainingParams.Coeff = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				osTrainingParams.Coeff = optimiseParamsMap[mapOffset + 1].second;
			}
		}

		template <typename SampleType>
		size_t SigmoidKernel<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size() * cvTrainingParams.CoeffToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename SampleType>
		DenseExtractor<SampleType>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename SampleType>
		DenseExtractor<SampleType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename SampleType>
		DenseExtractor<SampleType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename SampleType>
		typename DenseExtractor<SampleType>::ExtractorFunctionType DenseExtractor<SampleType>::GetExtractor(OneShotTrainingParams const& osTrainingParams)
		{
			return ExtractorFunctionType();
		}

		template <typename SampleType> template <class RegressionType>
		static void DenseExtractor<SampleType>::IterateExtractorParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& extractorCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			regressionParamSets.emplace_back(regressionOneShotTrainingParams);
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void DenseExtractor<SampleType>::PackageParameters(size_t const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void DenseExtractor<SampleType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const mapOffset)
		{
		}

		template <typename SampleType> template <size_t TotalNumParams>
		static void DenseExtractor<SampleType>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset)
		{
		}

		template <typename SampleType>
		size_t DenseExtractor<SampleType>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return 1u;
		}
	}
}