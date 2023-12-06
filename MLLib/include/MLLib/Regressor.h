#pragma once
#include <exception>
#include <string>
#include <vector>
#include <memory>
#include <MLLib/RegressionTypes.h>
#include <MLLib/KernelTypes.h>
#include <MLLib/ModifierTypes.h>
#include <dlib/random_forest.h>

namespace Regressors
{
	enum class CrossValidationMetric
	{
		SumAbsoluteMax,
		SumAbsoluteMean,
		SumSquareMax,
		SumSquareMean,
		CovarianceCorrelation
	};

	class RegressorTrainer
	{
		template <class RegressorType, class... ModifierFunctionTypes>
		friend class impl;
		template <typename SampleType>
		friend class Regressor;
		friend class RegressionComponentBase;
	public:
		struct RegressorError : public dlib::error
		{
			RegressorError(const std::string& message) : dlib::error(message) {}
		};
	private:
		template <typename SampleType>
		class impl_base
		{
			template <typename SampleType2>
			friend void serialize(Regressor<SampleType2> const& item, std::ostream& out);
		public:
			typedef typename SampleType::type T;

			virtual T Predict(SampleType input) const = 0;
			virtual T const& GetTrainingError() const = 0;
			virtual typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& GetTrainedRegressorParams() const = 0;
			virtual constexpr size_t GetNumModifiers() const = 0;
			virtual typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const = 0;
			virtual RegressionTypes::RegressorTypes const& GetRegressorType() const = 0;
		protected:
			virtual void Serialize(std::ostream& out) const = 0;

			template <size_t I = 0, class... ModifierFunctionTypes>
			static void SerializeModifiers(std::tuple<ModifierFunctionTypes...> const&, std::ostream& out);

			template <size_t I = 0, class... ModifierFunctionTypes>
			static typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetModifierTrainingParams(size_t const index,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions);

			template <size_t I = 0, class... ModifierFunctionTypes>
			static void ApplyModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
				SampleType& input);
		};

		template <class RegressionType, class... ModifierOneShotParamsTypes>
		static typename RegressionType::SampleType::type CrossValidate(const std::vector<typename RegressionType::SampleType>& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::tuple<ModifierOneShotParamsTypes...> const& modifierOneShotTrainingParams);

		template <class RegressionType, class... ModifierOneShotTrainingTypes>
		static void CrossValidateTrainingParameterSets(std::vector<typename RegressionType::SampleType> const& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			std::vector<typename RegressionType::OneShotTrainingParams> const& regressionTrainingParamsToTry,
			std::vector<std::tuple<ModifierOneShotTrainingTypes...>>& modifierTrainingParamsToTry,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::vector<std::pair<std::pair<size_t, size_t>, typename RegressionType::SampleType::type>>& allCrossValidatedRegressors,
			dlib::thread_pool& tp);

		template <size_t I, class... ModifierFindMinGlobalTrainingTypes>
		static constexpr size_t GetNumModifierParams();

		template <typename T, size_t TotalNumParams>
		static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const offset);

		template <typename T, size_t TotalNumParams, class... ModifierFindMinGlobalTrainingTypes, class ModifierFindMinGlobalTrainingType>
		static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			size_t const offset,
			ModifierFindMinGlobalTrainingType const& first,
			ModifierFindMinGlobalTrainingTypes const&... rest);

		template <typename T, size_t TotalNumParams>
		static void PackageModifierParams(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset);

		template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
		static void PackageModifierParams(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset,
			ModifierFindMinGlobalTrainingType const& first,
			ModifierFindMinGlobalTrainingTypes const&... rest);

		template <typename T, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
		static void UnpackModifierParams(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
			col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t const mapOffset,
			size_t& paramsOffset);

		template <typename T, size_t TotalNumParams>
		static void ProcessModifierFindMinGlobalParameterPack(std::array<std::pair<bool, T>, TotalNumParams> optimiseParamsMap);

		template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
		static void ProcessModifierFindMinGlobalParameterPack(std::array<std::pair<bool, T>, TotalNumParams> optimiseParamsMap,
			ModifierFindMinGlobalTrainingType const& first,
			ModifierFindMinGlobalTrainingTypes const&... rest);

		template <size_t I = 0, class... ModifierCrossValidationTrainingTypes>
		static size_t NumModifiersCrossValidationPermutations(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams);

	public:

		template <class RegressionType, class... ModifierOneShotTrainingTypes>
		static impl<RegressionType, typename ModifierOneShotTrainingTypes::ModifierType::ModifierFunction...> TrainRegressorOneShot(const std::vector<typename RegressionType::SampleType>& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			std::vector<typename RegressionType::SampleType::type>& diagnostics,
			typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
			ModifierOneShotTrainingTypes const&... modifiersOneShotTrainingPack);

		template <class RegressionType, class... ModifierCrossValidationTrainingTypes>
		static impl<RegressionType, typename ModifierCrossValidationTrainingTypes::ModifierType::ModifierFunction...> TrainRegressorCrossValidation(const std::vector<typename RegressionType::SampleType>& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			size_t const numThreads,
			std::vector<typename RegressionType::SampleType::type>& diagnostics,
			typename RegressionType::CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			ModifierCrossValidationTrainingTypes const&... modifiersCrossValidationTrainingPack);

		template <class RegressionType, class... ModifierFindMinGlobalTrainingTypes>
		static impl<RegressionType, typename ModifierFindMinGlobalTrainingTypes::ModifierType::ModifierFunction...> TrainRegressorFindMinGlobal(const std::vector<typename RegressionType::SampleType>& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			typename RegressionType::SampleType::type const& optimisationTolerance,
			size_t const numThreads,
			size_t const maxNumCalls,
			std::vector<typename RegressionType::SampleType::type>& diagnostics,
			typename RegressionType::FindMinGlobalTrainingParams const& regressionFindMinGlobalTrainingParams,
			ModifierFindMinGlobalTrainingTypes const&... modifiersFindMinGlobalTrainingPack);
	};

	template <typename SampleType>
	class Regressor
	{
	private:
		std::unique_ptr<RegressorTrainer::impl_base<SampleType>> m_impl;

	public:
		typedef SampleType SampleType;
		typedef typename SampleType::type T;
		friend class RegressionTypes::RegressionComponentBase;

		Regressor();

		~Regressor();

		Regressor(const Regressor& other);

		Regressor& operator= (Regressor const& other);

		Regressor(Regressor&& other) noexcept;

		Regressor& operator= (Regressor&& other) noexcept;

		template <class RegressionType, class... ModifierFunctionTypes>
		Regressor(impl<RegressionType, ModifierFunctionTypes...> const& regressor);

		template <class RegressionType, class... ModifierFunctionTypes>
		Regressor(impl<RegressionType, ModifierFunctionTypes...>&& regressor);

		T Predict(SampleType const& input) const;
		typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& GetTrainedRegressorParams() const;
		constexpr size_t GetNumModifiers() const;
		typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const;
		T const& GetTrainingError() const;
		RegressionTypes::RegressorTypes const& GetRegressorType() const;

		template <typename SampleType2>
		friend void serialize(Regressor<SampleType2> const& item, std::ostream& out);

		template <typename SampleType2>
		friend void deserialize(Regressor<SampleType2>& item, std::istream& in);

		template <class RegressionType>
		friend void DelegateDeserialize(Regressor<typename RegressionType::SampleType>& item, std::istream& in);
		
		template <class RegressionType, size_t I = 0, class... ModifierFunctionTypes>
		friend void DeserializeModifiers(size_t const numModifiers,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
			typename RegressionType::OneShotTrainingParams const& regressionParams,
			typename RegressionType::DecisionFunction const& function,
			typename RegressionType::SampleType::type const& trainingError,
			Regressor<typename RegressionType::SampleType>& item,
			std::istream& in);
	};

	template <class RegressionType, class... ModifierFunctionTypes>
	class impl : public RegressorTrainer::impl_base<typename RegressionType::SampleType>
	{
	public:
		static size_t const MaxNumModifiers;

	private:
		typedef typename RegressionType::SampleType SampleType;
		typedef typename RegressionType::SampleType::type T;
		typename T TrainingError;
		typename RegressionType::OneShotTrainingParams TrainedRegressorParams;
		typename RegressionType::DecisionFunction Function;
		std::tuple<ModifierFunctionTypes...> ModifierFunctions;

	public:
		T Predict(SampleType input) const override;
		typename RegressionType::OneShotTrainingParams const& GetTrainedRegressorParams() const override;
		constexpr size_t GetNumModifiers() const override;
		typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const override;
		T const& GetTrainingError() const override;
		RegressionTypes::RegressorTypes const& GetRegressorType() const override;

		impl(typename RegressionType::DecisionFunction const& function,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
			T const& trainingError,
			typename RegressionType::OneShotTrainingParams const& regressorTrainingParams);

		impl();

		template <class RegressionType2, class... ModifierFunctionTypes2>
		friend void serialize(impl<RegressionType2, ModifierFunctionTypes2... > const& item, std::ostream& out);

		template <class RegressionType2, class... ModifierFunctionTypes2>
		friend void deserialize(impl<RegressionType2, ModifierFunctionTypes2...>& item, std::istream& in);

		template <typename SampleType2>
		friend void serialize(Regressor<SampleType2> const& item, std::ostream& out);

		template <typename SampleType2>
		friend void deserialize(Regressor<SampleType2>& item, std::istream& in);

	private:
		void Serialize(std::ostream& out) const override;
	};

	template <class RegressorType, class... ModifierFunctionTypes>
	size_t const impl<RegressorType, ModifierFunctionTypes...>::MaxNumModifiers = 5ull;
}

#include "impl/Regressor.hpp"