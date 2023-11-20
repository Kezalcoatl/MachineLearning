#pragma once
#include <MLLib/TypeDefinitions.h>
#include <dlib/svm.h>
#include <dlib/random_forest.h>

namespace Regressors
{
	namespace KernelTypes
	{
		template <typename T>
		class LinearKernel
		{
		public:
			typedef T T;
			typedef typename dlib::linear_kernel<col_vector<T>> KernelFunctionType;

			LinearKernel() = delete;

			static size_t const NumKernelParams;

			struct OneShotTrainingParams
			{
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
				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				FindMinGlobalTrainingParams();
			};

			static KernelFunctionType GetKernel(OneShotTrainingParams const& osParams);

			template <class RegressionType>
			static void IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(unsigned const mapOffset,
				col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
				unsigned const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		class PolynomialKernel
		{
		public:
			typedef T T;
			typedef typename dlib::polynomial_kernel<col_vector<T>> KernelFunctionType;

			PolynomialKernel() = delete;

			static size_t const NumKernelParams;

			struct OneShotTrainingParams
			{
				T Gamma;
				T Coeff;
				T Degree;

				OneShotTrainingParams();

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.Gamma, out);
					dlib::serialize(item.Coeff, out);
					dlib::serialize(item.Degree, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.Gamma, in);
					dlib::deserialize(item.Coeff, in);
					dlib::deserialize(item.Degree, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<T> GammaToTry;
				std::vector<T> CoeffToTry;
				std::vector<T> DegreeToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				T LowerGamma;
				T LowerCoeff;
				T LowerDegree;
				T UpperGamma;
				T UpperCoeff;
				T UpperDegree;

				FindMinGlobalTrainingParams();
			};

			static KernelFunctionType GetKernel(OneShotTrainingParams const& osParams);

			template <class RegressionType>
			static void IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(unsigned const mapOffset,
				col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
				unsigned const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		class RadialBasisKernel
		{
		public:
			typedef T T;
			typedef typename dlib::radial_basis_kernel<col_vector<T>> KernelFunctionType;

			RadialBasisKernel() = delete;

			static size_t const NumKernelParams;

			struct OneShotTrainingParams
			{
				T Gamma;

				OneShotTrainingParams();

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.Gamma, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.Gamma, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<T> GammaToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				T LowerGamma;
				T UpperGamma;

				FindMinGlobalTrainingParams();
			};

			static KernelFunctionType GetKernel(OneShotTrainingParams const& osParams);

			template <class RegressionType>
			static void IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(unsigned const mapOffset,
				col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
				unsigned const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		class SigmoidKernel
		{
		public:
			typedef T T;
			typedef typename dlib::sigmoid_kernel<col_vector<T>> KernelFunctionType;

			SigmoidKernel() = delete;

			static size_t const NumKernelParams;

			struct OneShotTrainingParams
			{
				T Gamma;
				T Coeff;

				OneShotTrainingParams();

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.Gamma, out);
					dlib::serialize(item.Coeff, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.Gamma, in);
					dlib::deserialize(item.Coeff, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<T> GammaToTry;
				std::vector<T> CoeffToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				T LowerGamma;
				T LowerCoeff;
				T UpperGamma;
				T UpperCoeff;

				FindMinGlobalTrainingParams();
			};

			static KernelFunctionType GetKernel(OneShotTrainingParams const& osParams);

			template <class RegressionType>
			static void IterateKernelParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(unsigned const mapOffset,
				col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
				unsigned const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		/////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <typename T>
		class DenseExtractor
		{
		public:
			typedef T T;
			typedef typename dlib::dense_feature_extractor ExtractorFunctionType;

			DenseExtractor() = delete;

			static size_t const NumExtractorParams;

			struct OneShotTrainingParams
			{
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
				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				FindMinGlobalTrainingParams();
			};

			static ExtractorFunctionType GetExtractor(OneShotTrainingParams const& osParams);

			template <class RegressionType>
			static void IterateExtractorParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(unsigned const mapOffset,
				col_vector<T>& lowerBound,
				col_vector<T>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
				unsigned const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<T> const& vecParams,
				std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
				unsigned const mapOffset,
				unsigned& paramsOffset);

			static unsigned NumCrossValidationPermutations(CrossValidationTrainingParams const& cvTrainingParams);
		};

		template <typename T>
		size_t const LinearKernel<T>::NumKernelParams = 0ull;
		template <typename T>
		size_t const PolynomialKernel<T>::NumKernelParams = 3ull;
		template <typename T>
		size_t const RadialBasisKernel<T>::NumKernelParams = 1ull;
		template <typename T>
		size_t const SigmoidKernel<T>::NumKernelParams = 2ull;
		template <typename T>
		size_t const DenseExtractor<T>::NumExtractorParams = 0ull;
	}
}

#include "impl/KernelTypes.hpp"