#pragma once
#include <dlib/statistics.h>
#include <dlib/svm.h>
#include <dlib/threads.h>
#include <dlib/global_optimization.h>
#include <type_traits>

namespace Regressors
{
	template <class RegressionType, class... ModifierOneShotParamsTypes>
	static typename RegressionType::SampleType::type RegressorTrainer::CrossValidate(const std::vector<typename RegressionType::SampleType>& inputExamples,
		std::vector<typename RegressionType::SampleType::type> const& targetExamples,
		std::string const& randomSeed,
		typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
		size_t const numFolds,
		CrossValidationMetric const metric,
		std::tuple<ModifierOneShotParamsTypes...> const& modifierOneShotTrainingParams)
	{
		typedef typename RegressionType::T T;
		size_t const numExamples = inputExamples.size();
		size_t const numOrdinates = inputExamples.begin()->size();
		size_t const chunks = numExamples / numFolds;

		DLIB_ASSERT(numFolds > 0 && numFolds < numExamples,
			"Input parameter numFolds must be greater than zero and less than the provided number of examples.");

		dlib::running_stats<T> rs_abs;
		dlib::running_stats<T> rs_sq;
		dlib::running_scalar_covariance<T> rs_rc;

		for (size_t fold = 0; fold < numFolds; ++fold)
		{
			size_t const testStartIndex = fold * chunks;
			size_t const testEndIndex = fold == numFolds ? numExamples : (fold + 1) * chunks;
			size_t const numTestExamples = testEndIndex - testStartIndex;
			std::vector<col_vector<T>> foldTrainExamples(numExamples - numTestExamples, col_vector<T>(numOrdinates));
			std::vector<T> foldTrainTargets(numExamples - numTestExamples);
			std::vector<col_vector<T>> foldTestExamples(numTestExamples, col_vector<T>(numOrdinates));
			std::vector<T> foldTestTargets(numTestExamples);

			std::vector<size_t> randomIndices(numExamples);
			std::iota(randomIndices.begin(), randomIndices.end(), 0);
			dlib::rand rng(randomSeed);
			dlib::randomize_samples(randomIndices, rng);
			for (size_t i = 0; i < testStartIndex; ++i)
			{
				foldTrainExamples[i] = inputExamples[randomIndices[i]];
				foldTrainTargets[i] = targetExamples[randomIndices[i]];
			}
			for (size_t i = testStartIndex; i < testEndIndex; ++i)
			{
				foldTestExamples[i - testStartIndex] = inputExamples[randomIndices[i]];
				foldTestTargets[i - testStartIndex] = targetExamples[randomIndices[i]];
			}
			for (size_t i = testEndIndex; i < numExamples; ++i)
			{
				foldTrainExamples[i - numTestExamples] = inputExamples[randomIndices[i]];
				foldTrainTargets[i - numTestExamples] = targetExamples[randomIndices[i]];
			}

			std::tuple<typename ModifierOneShotParamsTypes::ModifierType::ModifierFunction...> modifierFunctions;
			std::vector<T> additionalDiagnostics;
			auto const predictor = RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(foldTrainExamples, 
				foldTrainTargets,
				regressionOneShotTrainingParams, 
				additionalDiagnostics,
				0.0,
				modifierOneShotTrainingParams,
				modifierFunctions);
			for (size_t i = 0; i < numTestExamples; ++i)
			{
				T result = predictor.Predict(foldTestExamples[i]);

				T diff = result - foldTestTargets[i];
				rs_abs.add(std::abs(diff));
				rs_sq.add(diff * diff);
				rs_rc.add(result, foldTestTargets[i]);
			}
		}

		switch (metric)
		{
		case CrossValidationMetric::SumSquareMax:
			return rs_sq.max();
		case CrossValidationMetric::SumSquareMean:
			return rs_sq.mean();
		case CrossValidationMetric::SumAbsoluteMax:
			return rs_abs.max();
		case CrossValidationMetric::SumAbsoluteMean:
			return rs_abs.mean();
		case CrossValidationMetric::CovarianceCorrelation:
			return rs_rc.correlation();
		default:
			throw RegressorError("Unrecognised cross-validation error metric.");
		}
	}

	template <class RegressionType, class... ModifierOneShotTrainingTypes>
	static impl<RegressionType, typename ModifierOneShotTrainingTypes::ModifierType::ModifierFunction...> RegressorTrainer::TrainRegressorOneShot(const std::vector<typename RegressionType::SampleType>& inputExamples,
		std::vector<typename RegressionType::SampleType::type> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		std::vector<typename RegressionType::SampleType::type>& diagnostics,
		typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
		ModifierOneShotTrainingTypes const&... modifiersOneShotTrainingPack)
	{
		static_assert(std::is_base_of<Regressors::RegressionTypes::RegressionComponentBase, RegressionType>::value, "RegressionType must be a derived type of RegressionTypes::RegressionComponentBase.");
		DLIB_ASSERT(dlib::is_learning_problem(inputExamples, targetExamples),
			"Bad input data.");

		std::tuple<ModifierOneShotTrainingTypes...> modifiersTrainingParams(modifiersOneShotTrainingPack...);
		auto const trainingError = CrossValidate<RegressionType>(inputExamples, targetExamples, randomSeed, regressionOneShotTrainingParams, numFolds, metric, modifiersTrainingParams);
		std::tuple<typename ModifierOneShotTrainingTypes::ModifierType::ModifierFunction...> modifierFunctions;
		return RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(inputExamples,
			targetExamples,
			regressionOneShotTrainingParams,
			diagnostics,
			trainingError,
			modifiersTrainingParams,
			modifierFunctions);
	}

	template <class RegressionType, class... ModifierOneShotTrainingTypes>
	static void RegressorTrainer::CrossValidateTrainingParameterSets(std::vector<typename RegressionType::SampleType> const& inputExamples,
			std::vector<typename RegressionType::SampleType::type> const& targetExamples,
			std::string const& randomSeed,
			std::vector<typename RegressionType::OneShotTrainingParams> const& regressionTrainingParamsToTry,
			std::vector<std::tuple<ModifierOneShotTrainingTypes...>>& modifierTrainingParamsToTry,
			size_t const numFolds,
			CrossValidationMetric const metric,
			std::vector<std::pair<std::pair<size_t, size_t>, typename RegressionType::SampleType::type>>& regressionModifierParamsTrainingError,
			dlib::thread_pool& tp)
	{
		typedef typename RegressionType::SampleType::type T;

		regressionModifierParamsTrainingError.resize(regressionTrainingParamsToTry.size() * modifierTrainingParamsToTry.size());
		for (size_t i = 0; i < regressionModifierParamsTrainingError.size(); ++i)
		{
			tp.add_task_by_value([&](size_t index)
				{
					size_t regressionParamsIndex = index % regressionTrainingParamsToTry.size();
					size_t modifiersParamsIndex = index / regressionTrainingParamsToTry.size();
					regressionModifierParamsTrainingError[index].first.first = regressionParamsIndex;
					regressionModifierParamsTrainingError[index].first.second = modifiersParamsIndex;
					regressionModifierParamsTrainingError[index].second = RegressorTrainer::template CrossValidate<RegressionType>(inputExamples,
						targetExamples,
						randomSeed,
						regressionTrainingParamsToTry[regressionParamsIndex],
						numFolds,
						metric,
						modifierTrainingParamsToTry[modifiersParamsIndex]);
				}, dlib::future<size_t>(i));
		}
		tp.wait_for_all_tasks();
	}

	template <size_t I, class... ModifierFindMinGlobalTrainingTypes>
	static constexpr size_t RegressorTrainer::GetNumModifierParams()
	{
		if constexpr (I == sizeof...(ModifierFindMinGlobalTrainingTypes))
		{
			return 0ull;
		}
		else
		{
			using ModifierType = typename std::tuple_element<I, std::tuple<ModifierFindMinGlobalTrainingTypes...>>::type::ModifierType;
			return ModifierType::NumModifierParams + GetNumModifierParams<I + 1, ModifierFindMinGlobalTrainingTypes...>();
		}
	}

	template <size_t I, class... ModifierCrossValidationTrainingTypes>
	size_t RegressorTrainer::NumModifiersCrossValidationPermutations(std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams)
	{
		if constexpr (sizeof... (ModifierCrossValidationTrainingTypes) == 0ull)
		{
			return 1ull;
		}
		else
		{
			using ModifierType = typename std::tuple_element<I, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
			if constexpr (I + 1 == sizeof...(ModifierCrossValidationTrainingTypes))
			{
				return ModifierType::NumCrossValidationPermutations(std::get<I>(modifierCrossValidationParams));
			}
			else
			{
				return ModifierType::NumCrossValidationPermutations(std::get<I>(modifierCrossValidationParams)) * NumModifiersCrossValidationPermutations<I + 1>(modifierCrossValidationParams);
			}
		}
	}

	template <class RegressionType, class...ModifierCrossValidationTrainingTypes>
	static impl<RegressionType, typename ModifierCrossValidationTrainingTypes::ModifierType::ModifierFunction...> RegressorTrainer::TrainRegressorCrossValidation(const std::vector<typename RegressionType::SampleType>& inputExamples,
		std::vector<typename RegressionType::SampleType::type> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		size_t const numThreads,
		std::vector<typename RegressionType::SampleType::type>& diagnostics,
		typename RegressionType::CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
		ModifierCrossValidationTrainingTypes const&... modifiersCrossValidationTrainingPack)
	{
		typedef typename RegressionType::SampleType::type T;
		static_assert(std::is_base_of<Regressors::RegressionTypes::RegressionComponentBase, RegressionType>::value);
		DLIB_ASSERT(dlib::is_learning_problem(inputExamples, targetExamples),
			"Bad input data.");

		std::tuple<ModifierCrossValidationTrainingTypes...> modifierCrossValidationParams(modifiersCrossValidationTrainingPack...);
		
		std::vector<typename RegressionType::OneShotTrainingParams> regressionParamsToTry;
		RegressionType::IterateRegressionParams(regressionCrossValidationTrainingParams, regressionParamsToTry);
		
		std::vector<std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::OneShotTrainingParams...>> modifierParamsToTry;
		ModifierTypes::ModifierComponentBase::IterateModifiers(modifierCrossValidationParams, modifierParamsToTry);

		dlib::thread_pool tp(numThreads);
		std::vector<std::pair<std::pair<size_t, size_t>, T>> regressorModifierParamsIndexTrainingError;

		CrossValidateTrainingParameterSets<RegressionType>(inputExamples, targetExamples, randomSeed, regressionParamsToTry, modifierParamsToTry, numFolds, metric, regressorModifierParamsIndexTrainingError, tp);
		size_t bestIndex = 0;
		for (size_t i = 0; i < regressorModifierParamsIndexTrainingError.size(); ++i)
		{
			if (regressorModifierParamsIndexTrainingError[i].second < regressorModifierParamsIndexTrainingError[bestIndex].second)
			{
				bestIndex = i;
			}
		}
		std::tuple<typename ModifierCrossValidationTrainingTypes::ModifierType::ModifierFunction...> modifierFunctions;
		return RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(inputExamples,
			targetExamples,
			regressionParamsToTry[regressorModifierParamsIndexTrainingError[bestIndex].first.first],
			diagnostics,
			regressorModifierParamsIndexTrainingError[bestIndex].second,
			modifierParamsToTry[regressorModifierParamsIndexTrainingError[bestIndex].first.second],
			modifierFunctions);
	}

	template <typename T, size_t TotalNumParams>
	static void RegressorTrainer::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
		size_t const offset)
	{
	}

	template <typename T, size_t TotalNumParams, class... ModifierFindMinGlobalTrainingTypes, class ModifierFindMinGlobalTrainingType>
	static void RegressorTrainer::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, 
		size_t const offset,
		ModifierFindMinGlobalTrainingType const& first,
		ModifierFindMinGlobalTrainingTypes const&... rest)
	{
		ModifierFindMinGlobalTrainingType::ModifierType::ConfigureModifierMapping(optimiseParamsMap, offset, first);
		RegressorTrainer::ConfigureModifierMapping<T>(optimiseParamsMap, offset + ModifierFindMinGlobalTrainingType::ModifierType::NumModifierParams, rest...);
	}

	template <typename T, size_t TotalNumParams>
	static void RegressorTrainer::PackageModifierParams(col_vector<T>& lowerBound,
		col_vector<T>& upperBound,
		std::vector<bool>& isIntegerParam,
		std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
		size_t const mapOffset,
		size_t& paramsOffset)
	{

	}

	template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
	static void RegressorTrainer::PackageModifierParams(col_vector<T>& lowerBound,
		col_vector<T>& upperBound,
		std::vector<bool>& isIntegerParam,
		std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
		size_t const mapOffset,
		size_t& paramsOffset,
		ModifierFindMinGlobalTrainingType const& first,
		ModifierFindMinGlobalTrainingTypes const&... rest)
	{
		ModifierFindMinGlobalTrainingType::ModifierType::PackageParameters(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, mapOffset, paramsOffset, first);
		RegressorTrainer::PackageModifierParams<T>(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, mapOffset + ModifierFindMinGlobalTrainingType::ModifierType::NumModifierParams, paramsOffset, rest...);
	}

	template <typename T, size_t TotalNumParams, class... ModifierOneShotTrainingParams>
	void RegressorTrainer::UnpackModifierParams(std::tuple<ModifierOneShotTrainingParams...>& modifierOneShotTrainingParams,
		col_vector<T> const& vecParams,
		std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
		size_t const mapOffset,
		size_t& paramsOffset)
	{
		if constexpr (sizeof...(ModifierOneShotTrainingParams) > 0)
		{
			using ModifierType = typename std::tuple_element<0, std::tuple<ModifierOneShotTrainingParams...>>::type::ModifierType;
			ModifierType::UnpackParameters<0>(modifierOneShotTrainingParams,
				vecParams,
				optimiseParamsMap,
				mapOffset,
				paramsOffset);
		}
	}

	template <typename T, size_t TotalNumParams>
	static void RegressorTrainer::ProcessModifierFindMinGlobalParameterPack(std::array<std::pair<bool, T>, TotalNumParams> optimiseParamsMap)
	{

	}

	template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
	static void RegressorTrainer::ProcessModifierFindMinGlobalParameterPack(std::array<std::pair<bool, T>, TotalNumParams> optimiseParamsMap,
		ModifierFindMinGlobalTrainingType const& first,
		ModifierFindMinGlobalTrainingTypes const&... rest)
	{

	}

	template <class RegressionType, class... ModifierFindMinGlobalTrainingTypes>
	static impl<RegressionType, typename ModifierFindMinGlobalTrainingTypes::ModifierType::ModifierFunction...> RegressorTrainer::TrainRegressorFindMinGlobal(const std::vector<typename RegressionType::SampleType>& inputExamples,
		std::vector<typename RegressionType::SampleType::type> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		typename RegressionType::SampleType::type const& optimisationTolerance,
		size_t const numThreads,
		size_t const maxNumCalls,
		std::vector<typename RegressionType::SampleType::type>& diagnostics,
		typename RegressionType::FindMinGlobalTrainingParams const& regressionFindMinGlobalTrainingParams,
		ModifierFindMinGlobalTrainingTypes const&... modifiersFindMinGlobalTrainingPack)
	{
		typedef typename RegressionType::SampleType::type T;
		static_assert(std::is_base_of<Regressors::RegressionTypes::RegressionComponentBase, RegressionType>::value);
		DLIB_ASSERT(dlib::is_learning_problem(inputExamples, targetExamples),
			"Bad input data.");

		constexpr size_t totalNumParams = RegressionType::NumTotalParams + GetNumModifierParams<0ull, ModifierFindMinGlobalTrainingTypes...>();
		std::array<std::pair<bool, T>, totalNumParams> optimiseParamsMap;
		RegressionType::ConfigureMapping(regressionFindMinGlobalTrainingParams, optimiseParamsMap);
		ConfigureModifierMapping<T>(optimiseParamsMap, RegressionType::NumTotalParams, modifiersFindMinGlobalTrainingPack...);

		const size_t numParamsToOptimise = std::accumulate(optimiseParamsMap.begin(), optimiseParamsMap.end(), 0ull, [](size_t sum, std::pair<bool, T> const& pair)
			{
				return pair.first ? sum + 1ull : sum;
			});
		DLIB_ASSERT(numParamsToOptimise > 0,
			"All parameters fixed - unable to run optimisation");
		col_vector<T> lowerBound(numParamsToOptimise);
		col_vector<T> upperBound(numParamsToOptimise);
		std::vector<bool> isIntegerParam(numParamsToOptimise);

		size_t paramsOffset = 0u;
		RegressionType::PackageParameters(lowerBound, upperBound, isIntegerParam, regressionFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		PackageModifierParams<T>(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, RegressionType::NumTotalParams, paramsOffset, modifiersFindMinGlobalTrainingPack...);

		dlib::thread_pool tp(numThreads);
		dlib::max_function_calls numCalls(maxNumCalls);
		auto findMinGlobalMetric = [&](col_vector<T> const& params)
		{
			size_t paramsOffset = 0u;
			typename RegressionType::OneShotTrainingParams const regressionParams(params, optimiseParamsMap, paramsOffset);

			std::tuple<typename ModifierFindMinGlobalTrainingTypes::ModifierType::OneShotTrainingParams...> modifierTrainingParams;
			UnpackModifierParams<T>(modifierTrainingParams, params, optimiseParamsMap, RegressionType::NumTotalParams, paramsOffset);
			try
			{
				return CrossValidate<RegressionType>(inputExamples, targetExamples, randomSeed, regressionParams, numFolds, metric, modifierTrainingParams);
			}
			catch (std::exception const& ex)
			{
				std::cout << ex.what() << std::endl;
			}
		};

		auto const result = dlib::find_min_global(tp, findMinGlobalMetric, lowerBound, upperBound, isIntegerParam, numCalls, optimisationTolerance);
		paramsOffset = 0u;
		typename RegressionType::OneShotTrainingParams optimisedRegressionParams(result.x, optimiseParamsMap, paramsOffset);

		std::tuple<typename ModifierFindMinGlobalTrainingTypes::ModifierType::OneShotTrainingParams...> optimisedModifierTrainingParams;
		UnpackModifierParams<T>(optimisedModifierTrainingParams, result.x, optimiseParamsMap, RegressionType::NumTotalParams, paramsOffset);

		std::tuple<typename ModifierFindMinGlobalTrainingTypes::ModifierType::ModifierFunction...> modifierFunctions;
		return RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(inputExamples,
			targetExamples,
			optimisedRegressionParams,
			diagnostics,
			result.y,
			optimisedModifierTrainingParams,
			modifierFunctions);
	}

	template <typename SampleType>
	Regressor<SampleType>::Regressor()
	{
		static_assert(std::is_floating_point<T>());
	}

	template <typename SampleType>
	Regressor<SampleType>::~Regressor()
	{
		m_impl.release();
	}

	template <typename SampleType>
	Regressor<SampleType>::Regressor(Regressor const& other)
	{
		this->operator=(other);
	}

	template <typename SampleType>
	Regressor<SampleType>& Regressor<SampleType>::operator=(Regressor const& other)
	{
		if (&other != this)
		{
			if (!other.m_impl)
			{
				m_impl.reset();
			}
			else
			{
				m_impl = std::unique_ptr<RegressorTrainer::impl_base<SampleType>>(other.m_impl.get());
			}
		}
		return *this;
	}

	template <typename SampleType>
	Regressor<SampleType>::Regressor(Regressor&& other) noexcept = default;

	template <typename SampleType>
	Regressor<SampleType>& Regressor<SampleType>::operator=(Regressor&& other) noexcept = default;

	template <typename SampleType> template <class RegressionType, class... ModifierFunctionTypes>
	Regressor<SampleType>::Regressor(impl<RegressionType, ModifierFunctionTypes...> const& regressor)
	{
		m_impl = std::make_unique<impl<RegressionType, ModifierFunctionTypes...>>(regressor);
	}

	template <typename SampleType> template <class RegressionType, class... ModifierFunctionTypes>
	Regressor<SampleType>::Regressor(impl<RegressionType, ModifierFunctionTypes...>&& regressor)
	{
		m_impl = std::make_unique<impl<RegressionType, ModifierFunctionTypes...>>(std::move(regressor));
	}

	template <typename SampleType>
	typename SampleType::type Regressor<SampleType>::Predict(SampleType const& input) const
	{
		return m_impl->Predict(input);
	}

	template <typename SampleType>
	RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& Regressor<SampleType>::GetTrainedRegressorParams() const
	{
		return m_impl->GetTrainedRegressorParams();
	}

	template <typename SampleType>
	constexpr size_t Regressor<SampleType>::GetNumModifiers() const
	{
		return m_impl->GetNumModifiers();
	}

	template <typename SampleType>
	ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& Regressor<SampleType>::GetTrainedModifierParams(size_t const index) const
	{
		return m_impl->GetTrainedModifierParams(index);
	}

	template <typename SampleType>
	typename SampleType::type const& Regressor<SampleType>::GetTrainingError() const
	{
		return m_impl->GetTrainingError();
	}

	template <typename SampleType>
	RegressionTypes::RegressorTypes const& Regressor<SampleType>::GetRegressorType() const
	{
		return m_impl->GetRegressorType();
	}

	template <typename SampleType>
	void serialize(Regressor<SampleType> const& item, std::ostream& out)
	{
		bool isNonEmpty = item.m_impl != nullptr;
		dlib::serialize(isNonEmpty, out);
		if (isNonEmpty)
		{
			serialize(item.GetRegressorType(), out);
			item.m_impl->Serialize(out);
		}
	}

	template <class RegressionType, size_t I, class... ModifierFunctionTypes>
	void DeserializeModifiers(size_t const numModifiers,
		std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		typename RegressionType::OneShotTrainingParams const& regressionParams,
		typename RegressionType::DecisionFunction const& function,
		typename RegressionType::SampleType::type const& trainingError,
		Regressor<typename RegressionType::SampleType>& item,
		std::istream& in)
	{
		if constexpr (I == impl<RegressionType, ModifierFunctionTypes...>::MaxNumModifiers)
		{
			throw RegressorTrainer::RegressorError("Modifier overflow.");
		}
		else if (I == numModifiers)
		{
			using RegressorType = impl<RegressionType, ModifierFunctionTypes...>;
			item.m_impl = std::make_unique<RegressorType>(function, modifierFunctions, trainingError, regressionParams);
		}
		else
		{
			ModifierTypes::ModifierTypes modifierType = ModifierTypes::ModifierTypes::MAX_NUMBER_OF_ModifierTypes;
			deserialize(modifierType, in);
			switch (modifierType)
			{
			case ModifierTypes::ModifierTypes::normaliser:
			{
				typename ModifierTypes::NormaliserModifier<typename RegressionType::SampleType>::ModifierFunction modifierFunction;
				deserialize(modifierFunction, in);
				DeserializeModifiers<RegressionType, I + 1>(numModifiers,
					std::tuple_cat(modifierFunctions, std::make_tuple(modifierFunction)),
					regressionParams,
					function,
					trainingError,
					item,
					in);
				break;

			}
			case ModifierTypes::ModifierTypes::inputPCA:
			{
				typename ModifierTypes::InputPCAModifier<typename RegressionType::SampleType>::ModifierFunction modifierFunction;
				deserialize(modifierFunction, in);
				DeserializeModifiers<RegressionType, I + 1>(numModifiers,
					std::tuple_cat(modifierFunctions, std::make_tuple(modifierFunction)),
					regressionParams,
					function,
					trainingError,
					item,
					in);
				break;
			}
			case ModifierTypes::ModifierTypes::featureSelection:
			{
				typename ModifierTypes::FeatureSelectionModifier<typename RegressionType::SampleType>::ModifierFunction modifierFunction;
				deserialize(modifierFunction, in);
				DeserializeModifiers<RegressionType, I + 1>(numModifiers,
					std::tuple_cat(modifierFunctions, std::make_tuple(modifierFunction)),
					regressionParams,
					function,
					trainingError,
					item,
					in);
				break;
			}
			default:
				throw RegressorTrainer::RegressorError("Unrecognised modifier type.");
			}
		}
	}

	template <class RegressionType>
	void DelegateDeserialize(Regressor<typename RegressionType::SampleType>& item, std::istream& in)
	{
		static_assert(std::is_base_of<RegressionTypes::RegressionComponentBase, RegressionType>::value);
		typename RegressionType::OneShotTrainingParams params;
		deserialize(params, in);
		typename RegressionType::DecisionFunction function;
		deserialize(function, in);
		typename RegressionType::SampleType::type trainingError(std::numeric_limits<typename RegressionType::SampleType::type>::max());
		dlib::deserialize(trainingError, in);
		size_t numModifiers = 0u;
		dlib::deserialize(numModifiers, in);
		std::tuple<> modifiers;
		DeserializeModifiers<RegressionType, 0>(numModifiers,
			modifiers,
			params,
			function,
			trainingError,
			item,
			in);
	}

	template <typename SampleType>
	void deserialize(Regressor<SampleType>& item, std::istream& in)
	{
		bool isNonEmpty = false;
		dlib::deserialize(isNonEmpty, in);
		if (isNonEmpty)
		{
			auto regressorTypeEnum = RegressionTypes::RegressorTypes::MAX_NUMBER_OF_RegressorTypes;
			deserialize(regressorTypeEnum, in);

			switch (regressorTypeEnum)
			{
			case RegressionTypes::RegressorTypes::LinearKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::LinearKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::PolynomialKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::PolynomialKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::RadialBasisKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::RadialBasisKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::SigmoidKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::SigmoidKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::LinearSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::LinearKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::PolynomialSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::PolynomialKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::RadialBasisSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::RadialBasisKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::SigmoidSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::SigmoidKernel<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::DenseRandomForestRegression:
			{
				typedef RegressionTypes::RandomForestRegression<KernelTypes::DenseExtractor<SampleType>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			default:
				throw RegressorTrainer::RegressorError("Unrecognised regressor type.");
			}
		}
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionType::SampleType::type impl<RegressionType, ModifierFunctionTypes...>::Predict(typename RegressionType::SampleType input) const
	{
		impl_base<typename RegressionType::SampleType>::ApplyModifiers(ModifierFunctions, input);
		return Function(input);
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionType::OneShotTrainingParams const& impl<RegressionType, ModifierFunctionTypes...>::GetTrainedRegressorParams() const
	{
		return TrainedRegressorParams;
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	constexpr size_t impl<RegressionType, ModifierFunctionTypes...>::GetNumModifiers() const
	{
		return std::tuple_size<decltype(ModifierFunctions)>::value;
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& impl<RegressionType, ModifierFunctionTypes...>::GetTrainedModifierParams(size_t const index) const
	{
		return impl_base<typename RegressionType::SampleType>::GetModifierTrainingParams(index, ModifierFunctions);
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionType::SampleType::type const& impl<RegressionType, ModifierFunctionTypes...>::GetTrainingError() const
	{
		return TrainingError;
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	RegressionTypes::RegressorTypes const& impl<RegressionType, ModifierFunctionTypes...>::GetRegressorType() const
	{
		return RegressionType::RegressorTypeEnum;
	}

	template <typename SampleType> template <size_t I, class... ModifierFunctionTypes>
	void RegressorTrainer::impl_base<SampleType>::SerializeModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions, std::ostream& out)
	{
		if constexpr (sizeof... (ModifierFunctionTypes) == I || sizeof... (ModifierFunctionTypes) == 0)
		{
			return;
		}
		else
		{
			using ModifierType = typename std::tuple_element<I, std::tuple<ModifierFunctionTypes...>>::type::ModifierType;
			serialize(ModifierType::ModifierTypeEnum, out);
			auto const& modifierFunction = std::get<I>(modifierFunctions);
			serialize(modifierFunction, out);
			SerializeModifiers<I + 1>(modifierFunctions, out);
		}
	}

	template <typename SampleType> template <size_t I, class... ModifierFunctionTypes>
	typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& RegressorTrainer::impl_base<SampleType>::GetModifierTrainingParams(size_t const index, std::tuple<ModifierFunctionTypes...> const& modifierFunctions)
	{
		if constexpr (sizeof... (ModifierFunctionTypes) == 0 || sizeof... (ModifierFunctionTypes) == I)
		{
			throw RegressorTrainer::RegressorError("modifierFunctions tuple access out of bounds.");
		}
		else if (I == index)
		{
			return std::get<I>(modifierFunctions).TrainingParams;
		}
		else
		{
			return GetModifierTrainingParams<I + 1>(index, modifierFunctions);
		}
	}

	template <typename SampleType> template <size_t I, class... ModifierFunctionTypes>
	static void RegressorTrainer::impl_base<SampleType>::ApplyModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		typename SampleType& input)
	{
		if constexpr (I == sizeof...(ModifierFunctionTypes))
		{
			return;
		}
		else
		{
			auto const& function = std::get<I>(modifierFunctions);
			function(input);
			ApplyModifiers<I + 1>(modifierFunctions, input);
		}
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	impl<RegressionType, ModifierFunctionTypes...>::impl(typename RegressionType::DecisionFunction const& function,
		std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		typename T const& trainingError,
		typename RegressionType::OneShotTrainingParams const& regressorTrainingParams) :
		TrainingError(trainingError), TrainedRegressorParams(regressorTrainingParams), Function(function), ModifierFunctions(modifierFunctions)
	{
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	impl<RegressionType, ModifierFunctionTypes...>::impl() : TrainingError(std::numeric_limits<T>::max())
	{
	}

	template <class RegressionType2, class... ModifierFunctionTypes2>
	void serialize(impl<RegressionType2, ModifierFunctionTypes2... > const& item, std::ostream& out)
	{
		serialize(item.TrainedRegressorParams, out);
		serialize(item.Function, out);
		dlib::serialize(item.TrainingError, out);
		dlib::serialize(item.ModifierFunctions, out);
	}

	template <class RegressionType2, class... ModifierFunctionTypes2>
	void deserialize(impl<RegressionType2, ModifierFunctionTypes2...>& item, std::istream& in)
	{
		deserialize(item.TrainedRegressorParams, in);
		deserialize(item.Function, in);
		dlib::deserialize(item.TrainingError, in);
		dlib::deserialize(item.ModifierFunctions, in);
	}

	template <class RegressionType, class... ModifierFunctionTypes>
	void impl<RegressionType, ModifierFunctionTypes...>::Serialize(std::ostream& out) const
	{
		serialize(TrainedRegressorParams, out);
		serialize(Function, out);
		dlib::serialize(TrainingError, out);
		dlib::serialize(std::tuple_size<decltype(ModifierFunctions)>::value, out);
		RegressorTrainer::impl_base<typename RegressionType::SampleType>::SerializeModifiers(ModifierFunctions, out);
	}
}