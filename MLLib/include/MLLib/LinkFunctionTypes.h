#pragma once
#include <MLLib/TypeDefinitions.h>

namespace Regressors
{
	namespace LinkFunctionTypes
	{
		template <class KernelType>
		class LogitLinkFunction
		{
		public:
			typedef KernelType KernelType;
			typedef typename KernelType::KernelFunctionType KernelFunctionType;
			typedef typename KernelType::T ScalarType;
			typedef typename KernelType::SampleType SampleType;

			LogitLinkFunction() = delete;

			static size_t const NumLinkFunctionParams;

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

			struct LinkFunction
			{
				typedef LogitLinkFunction LinkFunctionType;

				OneShotTrainingParams TrainingParams;

				ScalarType operator()(ScalarType const& input) const;

				friend void serialize(LinkFunction const& item, std::ostream& out)
				{
					serialize(item.TrainingParams, out);
				}

				friend void deserialize(LinkFunction& item, std::istream& in)
				{
					deserialize(item.TrainingParams, in);
				}
			};

			static LinkFunction Train(OneShotTrainingParams const& osParams,
				col_vector<ScalarType> const& inputExamples,
				col_vector<ScalarType> const& targetExamples);

			template <class RegressionType>
			static void IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
				typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(size_t const mapOffset,
				col_vector<ScalarType>& lowerBound,
				col_vector<ScalarType>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
				size_t const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<ScalarType> const& vecParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <class KernelType>
		class FourierLinkFunction
		{
		public:
			typedef KernelType KernelType;
			typedef typename KernelType::KernelFunctionType KernelFunctionType;
			typedef typename KernelType::T ScalarType;
			typedef typename KernelType::SampleType SampleType;

			FourierLinkFunction() = delete;

			static size_t const NumLinkFunctionParams;

			struct OneShotTrainingParams
			{
				size_t NumTerms;

				OneShotTrainingParams();

				friend void serialize(OneShotTrainingParams const& item, std::ostream& out)
				{
					dlib::serialize(item.NumTerms, out);
				}

				friend void deserialize(OneShotTrainingParams& item, std::istream& in)
				{
					dlib::deserialize(item.NumTerms, in);
				}
			};

			struct CrossValidationTrainingParams
			{
				std::vector<size_t> NumTermsToTry;

				CrossValidationTrainingParams();
			};

			struct FindMinGlobalTrainingParams
			{
				size_t LowerNumTerms;
				size_t UpperNumTerms;

				FindMinGlobalTrainingParams();
			};

			struct LinkFunction
			{
				typedef FourierLinkFunction LinkFunctionType;

				ScalarType Bias;
				std::vector<std::pair<ScalarType, ScalarType>> Coefficients;
				ScalarType Min;
				ScalarType Max;
				ScalarType MinStep;
				OneShotTrainingParams TrainingParams;

				ScalarType operator()(ScalarType const& input) const;

				ScalarType Theta(ScalarType const& input) const;

				friend void serialize(LinkFunction const& item, std::ostream& out)
				{
					dlib::serialize(item.Bias, out);
					dlib::serialize(item.Coefficients, out);
					dlib::serialize(item.Min, out);
					dlib::serialize(item.Max, out);
					dlib::serialize(item.MinStep, out);
					serialize(item.TrainingParams, out);
				}

				friend void deserialize(LinkFunction& item, std::istream& in)
				{
					dlib::deserialize(item.Bias, in);
					dlib::deserialize(item.Coefficients, in);
					dlib::deserialize(item.Min, in);
					dlib::deserialize(item.Max, in);
					dlib::deserialize(item.MinStep, in);
					deserialize(item.TrainingParams, in);
				}
			};

			static LinkFunction Train(OneShotTrainingParams const& osParams,
				col_vector<ScalarType> const& inputExamples,
				col_vector<ScalarType> const& targetExamples);

			template <class RegressionType>
			static void IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
				typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(size_t const mapOffset,
				col_vector<ScalarType>& lowerBound,
				col_vector<ScalarType>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
				size_t const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<ScalarType> const& vecParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <class KernelType>
		class LagrangeLinkFunction
		{
		public:
			typedef KernelType KernelType;
			typedef typename KernelType::KernelFunctionType KernelFunctionType;
			typedef typename KernelType::T ScalarType;
			typedef typename KernelType::SampleType SampleType;

			LagrangeLinkFunction() = delete;

			static size_t const NumLinkFunctionParams;

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

			struct LinkFunction
			{
				typedef LagrangeLinkFunction LinkFunctionType;
				std::vector<ScalarType> Weights;
				std::vector<ScalarType> Roots;
				std::vector<ScalarType> Anchors;
				OneShotTrainingParams TrainingParams;

				ScalarType operator()(ScalarType const& input) const;

				friend void serialize(LinkFunction const& item, std::ostream& out)
				{
					dlib::serialize(item.Weights, out);
					dlib::serialize(item.Roots, out);
					dlib::serialize(item.Anchors, out);
					serialize(item.TrainingParams, out);
				}

				friend void deserialize(LinkFunction& item, std::istream& in)
				{
					dlib::deserialize(item.Weights, in);
					dlib::deserialize(item.Roots, in);
					dlib::deserialize(item.Anchors, in);
					deserialize(item.TrainingParams, in);
				}
			};

			static LinkFunction Train(OneShotTrainingParams const& osParams,
				col_vector<ScalarType> const& inputExamples,
				col_vector<ScalarType> const& targetExamples);

			template <class RegressionType>
			static void IterateLinkFunctionParams(typename RegressionType::OneShotTrainingParams& regressionOneShotTrainingParams,
				CrossValidationTrainingParams const& linkFunctionCrossValidationTrainingParams,
				typename RegressionType::KernelType::CrossValidationTrainingParams const& kernelCrossValidationTrainingParams,
				std::vector<typename RegressionType::OneShotTrainingParams>& regressionParamSets);

			template <size_t TotalNumParams>
			static void PackageParameters(size_t const mapOffset,
				col_vector<ScalarType>& lowerBound,
				col_vector<ScalarType>& upperBound,
				std::vector<bool>& isIntegerParam,
				FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t& paramsOffset);

			template <size_t TotalNumParams>
			static void ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams>& optimiseParamsMap,
				size_t const mapOffset);

			template <size_t TotalNumParams>
			static void UnpackParameters(OneShotTrainingParams& osTrainingParams,
				col_vector<ScalarType> const& vecParams,
				std::array<std::pair<bool, ScalarType>, TotalNumParams> const& optimiseParamsMap,
				size_t const mapOffset,
				size_t& paramsOffset);
		};

		template <class KernelType>
		size_t const LogitLinkFunction<KernelType>::NumLinkFunctionParams = 0ull;

		template <class KernelType>
		size_t const FourierLinkFunction<KernelType>::NumLinkFunctionParams = 1ull;

		template <class KernelType>
		size_t const LagrangeLinkFunction<KernelType>::NumLinkFunctionParams = 0ull;
	}
}

#include "impl/LinkFunctionTypes.hpp"