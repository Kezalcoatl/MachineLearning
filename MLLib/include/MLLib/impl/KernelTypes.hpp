#pragma once

namespace Regressors
{
	namespace KernelTypes
	{
		template <typename T>
		LinearKernel<T>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value, "T must be a floating point type.");
		}

		template <typename T>
		LinearKernel<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename T>
		LinearKernel<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename T>
		typename LinearKernel<T>::KernelFunctionType LinearKernel<T>::GetKernel(OneShotTrainingParams const& osParams)
		{
			return KernelFunctionType();
		}

		template <typename T> template <class RegressionType>
		static void LinearKernel<T>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			regressionParamSets.emplace_back(regressionOneShotTrainingParams);
		}

		template <typename T> template <size_t TotalNumParams>
		static void LinearKernel<T>::PackageParameters(unsigned const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned& paramsOffset)
		{
		}

		template <typename T> template <size_t TotalNumParams>
		static void LinearKernel<T>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const mapOffset)
		{
		}

		template <typename T> template <size_t TotalNumParams>
		static void LinearKernel<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
		{
		}

		template <typename T>
		unsigned LinearKernel<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return 1u;
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		PolynomialKernel<T>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0), Coeff(0.0), Degree(1.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		PolynomialKernel<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
			CoeffToTry = { temp.Coeff };
			DegreeToTry = { temp.Degree };
		}

		template <typename T>
		PolynomialKernel<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
			LowerCoeff = temp.Coeff;
			UpperCoeff = temp.Coeff;
			LowerDegree = temp.Degree;
			UpperDegree = temp.Degree;
		}

		template <typename T>
		typename PolynomialKernel<T>::KernelFunctionType PolynomialKernel<T>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma, osTrainingParams.Coeff, osTrainingParams.Degree);
		}

		template <typename T> template <class RegressionType>
		static void PolynomialKernel<T>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
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

		template <typename T> template <size_t TotalNumParams>
		static void PolynomialKernel<T>::PackageParameters(unsigned const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned& paramsOffset)
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

		template <typename T> template <size_t TotalNumParams>
		static void PolynomialKernel<T>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
			optimiseParamsMap[mapOffset + 1].first = fmgTrainingParams.LowerCoeff != fmgTrainingParams.UpperCoeff;
			optimiseParamsMap[mapOffset + 1].second = fmgTrainingParams.LowerCoeff;
			optimiseParamsMap[mapOffset + 2].first = fmgTrainingParams.LowerDegree != fmgTrainingParams.UpperDegree;
			optimiseParamsMap[mapOffset + 2].second = fmgTrainingParams.LowerDegree;
		}

		template <typename T> template <size_t TotalNumParams>
		static void PolynomialKernel<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
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

		template <typename T>
		unsigned PolynomialKernel<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size() * cvTrainingParams.CoeffToTry.size() * cvTrainingParams.DegreeToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		RadialBasisKernel<T>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		RadialBasisKernel<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
		}

		template <typename T>
		RadialBasisKernel<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
		}

		template <typename T>
		typename RadialBasisKernel<T>::KernelFunctionType RadialBasisKernel<T>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma);
		}

		template <typename T> template <class RegressionType>
		static void RadialBasisKernel<T>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
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

		template <typename T> template <size_t TotalNumParams>
		static void RadialBasisKernel<T>::PackageParameters(unsigned const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned& paramsOffset)
		{
			if (optimiseParamsMap[mapOffset].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerGamma;
				upperBound(paramsOffset) = fmgTrainingParams.UpperGamma;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
		}

		template <typename T> template <size_t TotalNumParams>
		static void RadialBasisKernel<T>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
		}

		template <typename T> template <size_t TotalNumParams>
		static void RadialBasisKernel<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
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

		template <typename T>
		unsigned RadialBasisKernel<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		SigmoidKernel<T>::OneShotTrainingParams::OneShotTrainingParams() : Gamma(1.0), Coeff(0.0)
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		SigmoidKernel<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			GammaToTry = { temp.Gamma };
			CoeffToTry = { temp.Coeff };
		}

		template <typename T>
		SigmoidKernel<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerGamma = temp.Gamma;
			UpperGamma = temp.Gamma;
			LowerCoeff = temp.Coeff;
			UpperCoeff = temp.Coeff;
		}

		template <typename T>
		typename SigmoidKernel<T>::KernelFunctionType SigmoidKernel<T>::GetKernel(OneShotTrainingParams const& osTrainingParams)
		{
			return KernelFunctionType(osTrainingParams.Gamma, osTrainingParams.Coeff);
		}

		template <typename T> template <class RegressionType>
		static void SigmoidKernel<T>::IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
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

		template <typename T> template <size_t TotalNumParams>
		static void SigmoidKernel<T>::PackageParameters(unsigned const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned& paramsOffset)
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

		template <typename T> template <size_t TotalNumParams>
		static void SigmoidKernel<T>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const mapOffset)
		{
			optimiseParamsMap[mapOffset].first = fmgTrainingParams.LowerGamma != fmgTrainingParams.UpperGamma;
			optimiseParamsMap[mapOffset].second = fmgTrainingParams.LowerGamma;
			optimiseParamsMap[mapOffset + 1].first = fmgTrainingParams.LowerCoeff != fmgTrainingParams.UpperCoeff;
			optimiseParamsMap[mapOffset + 1].second = fmgTrainingParams.LowerCoeff;
		}

		template <typename T> template <size_t TotalNumParams>
		static void SigmoidKernel<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
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

		template <typename T>
		unsigned SigmoidKernel<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return cvTrainingParams.GammaToTry.size() * cvTrainingParams.CoeffToTry.size();
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		DenseExtractor<T>::OneShotTrainingParams::OneShotTrainingParams()
		{
			static_assert(std::is_floating_point<T>::value);
		}

		template <typename T>
		DenseExtractor<T>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
		}

		template <typename T>
		DenseExtractor<T>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
		}

		template <typename T>
		typename DenseExtractor<T>::ExtractorFunctionType DenseExtractor<T>::GetExtractor(OneShotTrainingParams const& osTrainingParams)
		{
			return ExtractorFunctionType();
		}

		template <typename T> template <class RegressionType>
		static void DenseExtractor<T>::IterateExtractorParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
			CrossValidationTrainingParams const& extractorCrossValidationTrainingParams,
			std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets)
		{
			regressionParamSets.emplace_back(regressionOneShotTrainingParams);
		}

		template <typename T> template <size_t TotalNumParams>
		static void DenseExtractor<T>::PackageParameters(unsigned const mapOffset,
			col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned& paramsOffset)
		{
		}

		template <typename T> template <size_t TotalNumParams>
		static void DenseExtractor<T>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const mapOffset)
		{
		}

		template <typename T> template <size_t TotalNumParams>
		static void DenseExtractor<T>::UnpackParameters(OneShotTrainingParams& osTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset)
		{
		}

		template <typename T>
		unsigned DenseExtractor<T>::NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams)
		{
			return 1u;
		}
	}
}