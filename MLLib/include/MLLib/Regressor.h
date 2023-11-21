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
	public:
		struct RegressorError : public dlib::error
		{
			RegressorError(const std::string& message) : dlib::error(message) {}
		};
	private:
		template <class RegressionType, class... ModifierOneShotParamsTypes>
		static Regressor<typename RegressionType::T> CrossValidate(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			typename RegressionType::OneShotTrainingParams const& regressionTrainingParams,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::vector<typename RegressionType::T>& diagnostics,
			std::tuple<ModifierOneShotParamsTypes...> const& modifierOneShotParams);

		template <class RegressionType, class... ModifierCrossValidationTrainingTypes>
		static void ProcessModifierCrossValidationParameterPack(std::vector<col_vector<typename RegressionType::T>> const& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			std::vector<typename RegressionType::OneShotTrainingParams> const& regressionTrainingParamsToTry,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::vector<std::pair<Regressor<typename RegressionType::T>, std::vector<typename RegressionType::T>>>& allCrossValidatedRegressors,
			dlib::thread_pool& tp,
			std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams);

		template <class RegressionType, size_t I, class... ModifierFindMinGlobalTrainingTypes, class... ModifierOneShotTrainingTypes, size_t TotalNumParams>
		static Regressor<typename RegressionType::T> CreateModifiersAndCrossValidate(std::tuple<ModifierOneShotTrainingTypes...>& modifiersSoFar,
			col_vector<typename RegressionType::T> const& params,
			std::array<std::pair<bool, typename RegressionType::T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset,
			std::vector<col_vector<typename RegressionType::T>> const& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			typename RegressionType::OneShotTrainingParams const& regressionParams,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::vector<typename RegressionType::T>& diagnostics);

		template <size_t I, class... ModifierFindMinGlobalTrainingTypes>
		static constexpr size_t GetNumModifierParams();

		template <typename T, size_t TotalNumParams>
		static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const offset);

		template <typename T, size_t TotalNumParams, class... ModifierFindMinGlobalTrainingTypes, class ModifierFindMinGlobalTrainingType>
		static void ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
			unsigned const offset,
			ModifierFindMinGlobalTrainingType const& first,
			ModifierFindMinGlobalTrainingTypes const&... rest);

		template <typename T, size_t TotalNumParams>
		static void PackageModifierParams(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset);

		template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
		static void PackageModifierParams(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			unsigned const mapOffset,
			unsigned& paramsOffset,
			ModifierFindMinGlobalTrainingType const& first,
			ModifierFindMinGlobalTrainingTypes const&... rest);

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
		static Regressor<typename RegressionType::T> TrainRegressorOneShot(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			std::vector<typename RegressionType::T>& diagnostics,
			typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
			ModifierOneShotTrainingTypes const&... modifiersOneShotTrainingPack);

		template <class RegressionType, class... ModifierCrossValidationTrainingTypes>
		static Regressor<typename RegressionType::T> TrainRegressorCrossValidation(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			unsigned long const numThreads,
			std::vector<typename RegressionType::T>& diagnostics,
			typename RegressionType::CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			ModifierCrossValidationTrainingTypes const&... modifiersCrossValidationTrainingPack);

		template <class RegressionType, class... ModifierFindMinGlobalTrainingTypes>
		static Regressor<typename RegressionType::T> TrainRegressorFindMinGlobal(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
			std::vector<typename RegressionType::T> const& targetExamples,
			std::string const& randomSeed,
			CrossValidationMetric const metric,
			size_t const numFolds,
			typename RegressionType::T const& optimisationTolerance,
			unsigned long const numThreads,
			size_t const maxNumCalls,
			std::vector<typename RegressionType::T>& diagnostics,
			typename RegressionType::FindMinGlobalTrainingParams const& regressionFindMinGlobalTrainingParams,
			ModifierFindMinGlobalTrainingTypes const&... modifiersFindMinGlobalTrainingPack);
	};

	template <typename T>
	class Regressor
	{
	private:
		static size_t const MaxNumModifiers;
		class impl_base
		{
		public:
			virtual T Predict(col_vector<T> input) const = 0;
			virtual T const& GetTrainingError() const = 0;
			virtual typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& GetTrainedRegressorParams() const = 0;
			virtual constexpr size_t GetNumModifiers() const = 0;
			virtual typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const = 0;
			virtual RegressionTypes::RegressorTypes const& GetRegressorType() const = 0;
			virtual void Serialize(std::ostream& out) const = 0;
			template <size_t I = 0, class... ModifierFunctionTypes>
			static void SerializeModifiers(std::tuple<ModifierFunctionTypes...> const&, std::ostream& out);
		};

		template <class RegressionType, class... ModifierFunctionTypes>
		class impl : public Regressor<typename RegressionType::T>::impl_base
		{
		private:
			
			typedef typename RegressionType::T T;
			T TrainingError;
			typename RegressionType::OneShotTrainingParams TrainedRegressorParams;
			typename RegressionType::DecisionFunction Function;
			std::tuple<ModifierFunctionTypes...> ModifierFunctions;

		public:

			T Predict(col_vector<T> input) const override;
			typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& GetTrainedRegressorParams() const override;
			constexpr size_t GetNumModifiers() const override;
			typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const override;
			T const& GetTrainingError() const override;
			RegressionTypes::RegressorTypes const& GetRegressorType() const override;
			void Serialize(std::ostream& out) const override;

			impl(typename RegressionType::DecisionFunction const& function,
				std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
				typename RegressionType::T const& trainingError,
				typename RegressionType::OneShotTrainingParams const& regressorTrainingParams);

		private:
			template <size_t I = 0>
			static void ApplyModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
				col_vector<T>& input);

			template <size_t I = 0>
			static typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetModifierParams(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
				size_t const index);
		};

		std::unique_ptr<impl_base> m_impl;

	public:

		friend class RegressionTypes::RegressionComponentBase;

		Regressor();

		~Regressor();

		Regressor(const Regressor& other);

		Regressor& operator= (Regressor const& other);

		Regressor(Regressor&& other) noexcept;

		Regressor& operator= (Regressor&& other) noexcept;

		T Predict(col_vector<T> const& input) const;
		typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& GetTrainedRegressorParams() const;
		constexpr size_t GetNumModifiers() const;
		typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& GetTrainedModifierParams(size_t const index) const;
		T const& GetTrainingError() const;
		RegressionTypes::RegressorTypes const& GetRegresorType() const;

		template <typename T2>
		friend void serialize(Regressor<T2> const& item, std::ostream& out);

		template <typename T2>
		friend void deserialize(Regressor<T2>& item, std::istream& in);

		template <class RegressionType, typename T2>
		friend void DelegateDeserialize(Regressor<T2>& item, std::istream& in);
		
		template <class RegressionType, size_t I = 0, class... ModifierFunctionTypesSoFar>
		friend void DeserializeModifiers(unsigned const numModifiers,
			std::tuple<ModifierFunctionTypesSoFar...> const& modifierFunctions,
			typename RegressionType::OneShotTrainingParams const& regressionParams,
			typename RegressionType::DecisionFunction const& function,
			typename RegressionType::T const& trainingError,
			Regressor<typename RegressionType::T>& item,
			std::istream& in);
	};
	template <typename T>
	size_t const Regressor<T>::MaxNumModifiers = 5ull;
}

#include "impl/Regressor.hpp"