#pragma once
#include <dlib/statistics.h>
#include <dlib/svm.h>
#include <dlib/threads.h>
#include <dlib/global_optimization.h>
#include <type_traits>

namespace Regressors
{
	template <class RegressionType, class... ModifierOneShotParamsTypes>
	static Regressor<typename RegressionType::T> RegressorTrainer::CrossValidate(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
		std::vector<typename RegressionType::T> const& targetExamples,
		std::string const& randomSeed,
		typename RegressionType::OneShotTrainingParams const& regressionTrainingParams,
		size_t const numFolds,
		CrossValidationMetric const metric,
		std::vector<typename RegressionType::T>& diagnostics,
		std::tuple<ModifierOneShotParamsTypes...> const& modifierOneShotParams)
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

			std::vector<T> additionalDiagnostics;
			std::tuple<> modifierFunctions;
			auto predictor = Regressors::RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(foldTrainExamples, 
				foldTrainTargets,
				regressionTrainingParams, 
				additionalDiagnostics,
				0.0,
				modifierOneShotParams,
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

		T error = std::numeric_limits<T>::max();
		switch (metric)
		{
		case CrossValidationMetric::SumSquareMax:
			error = rs_sq.max();
			break;
		case CrossValidationMetric::SumSquareMean:
			error = rs_sq.mean();
			break;
		case CrossValidationMetric::SumAbsoluteMax:
			error = rs_abs.max();
			break;
		case CrossValidationMetric::SumAbsoluteMean:
			error = rs_abs.mean();
			break;
		case CrossValidationMetric::CovarianceCorrelation:
			error = rs_rc.correlation();
			break;
		default:
			throw RegressorError("Unrecognised cross-validation error metric.");
		}
		std::tuple<> modifierFunctions;
		return Regressors::RegressionTypes::RegressionComponentBase::template TrainModifiersAndRegressor<RegressionType>(inputExamples, targetExamples, regressionTrainingParams, diagnostics, error, modifierOneShotParams, modifierFunctions);
	}

	template <class RegressionType, class... ModifierOneShotTrainingTypes>
	static Regressor<typename RegressionType::T> RegressorTrainer::TrainRegressorOneShot(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
		std::vector<typename RegressionType::T> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		std::vector<typename RegressionType::T>& diagnostics,
		typename RegressionType::OneShotTrainingParams const& regressionOneShotTrainingParams,
		ModifierOneShotTrainingTypes const&... modifiersOneShotTrainingPack)
	{
		static_assert(std::is_base_of<Regressors::RegressionTypes::RegressionComponentBase, RegressionType>::value, "RegressionType must be a derived type of RegressionTypes::RegressionComponentBase.");
		DLIB_ASSERT(dlib::is_learning_problem(inputExamples, targetExamples),
			"Bad input data.");

		std::tuple<ModifierOneShotTrainingTypes...> modifiersTrainingParams(modifiersOneShotTrainingPack...);
		auto predictor = CrossValidate<RegressionType>(inputExamples, targetExamples, randomSeed, regressionOneShotTrainingParams, numFolds, metric, diagnostics, modifiersTrainingParams);
		diagnostics.resize(inputExamples.size());
		for (int i = 0; i < inputExamples.size(); ++i)
		{
			diagnostics[i] = predictor.Predict(inputExamples[i]) - targetExamples[i];
		}
		return predictor;
	}

	template <class RegressionType, class... ModifierCrossValidationTrainingTypes>
	static void RegressorTrainer::ProcessModifierCrossValidationParameterPack(std::vector<col_vector<typename RegressionType::T>> const& inputExamples,
		std::vector<typename RegressionType::T> const& targetExamples,
		std::string const& randomSeed,
		std::vector<typename RegressionType::OneShotTrainingParams> const& regressionTrainingParamsToTry,
		size_t const numFolds,
		CrossValidationMetric const metric,
		std::vector<std::pair<Regressor<typename RegressionType::T>, std::vector<typename RegressionType::T>>>& allCrossValidatedRegressors,
		dlib::thread_pool& tp,
		std::tuple<ModifierCrossValidationTrainingTypes...> const& modifierCrossValidationParams)
	{
		std::tuple<> iteratedModifiersOneShotParams;
		if constexpr (sizeof... (ModifierCrossValidationTrainingTypes) == 0)
		{
			for (dlib::future<size_t> i = 0; i < regressionTrainingParamsToTry.size(); ++i)
			{
				tp.add_task_by_value([&](size_t i)
					{
						allCrossValidatedRegressors[i].first = RegressorTrainer::template CrossValidate<RegressionType>(inputExamples,
							targetExamples,
							randomSeed,
							regressionTrainingParamsToTry[i],
							numFolds,
							metric,
							allCrossValidatedRegressors[i].second,
							iteratedModifiersOneShotParams);
					}, i);
			}
		}
		else
		{
			size_t start = 0ull;
			auto crossValidationCallback = [&](const auto& modsifierOneShotParams)
			{
				for (dlib::future<size_t> i = 0; i < regressionTrainingParamsToTry.size(); ++i)
				{
					tp.add_task_by_value([&](size_t i)
						{
							allCrossValidatedRegressors[start + i].first = RegressorTrainer::template CrossValidate<RegressionType>(inputExamples,
								targetExamples,
								randomSeed,
								regressionTrainingParamsToTry[i],
								numFolds,
								metric,
								allCrossValidatedRegressors[start + i].second,
								modsifierOneShotParams);
						}, i);
				}
				start += regressionTrainingParamsToTry.size();
			};
			using ModifierType = typename std::tuple_element<0, std::tuple<ModifierCrossValidationTrainingTypes...>>::type::ModifierType;
			ModifierType::template IterateModifierParams(crossValidationCallback, modifierCrossValidationParams, iteratedModifiersOneShotParams);
		}
		tp.wait_for_all_tasks();
	}

	template <class RegressionType, size_t I, class... ModifierFindMinGlobalTrainingTypes, class... ModifierOneShotTrainingTypes, size_t TotalNumParams>
	static Regressor<typename RegressionType::T> RegressorTrainer::CreateModifiersAndCrossValidate(std::tuple<ModifierOneShotTrainingTypes...>& modifierTrainingParams,
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
		std::vector<typename RegressionType::T>& diagnostics)
	{
		if constexpr (sizeof...(ModifierFindMinGlobalTrainingTypes) == I)
		{
			return CrossValidate<RegressionType>(inputExamples,
				targetExamples,
				randomSeed,
				regressionParams,
				numFolds,
				metric,
				diagnostics,
				modifierTrainingParams);
		}
		else
		{
			using ModifierType = typename std::tuple_element<I, std::tuple<ModifierFindMinGlobalTrainingTypes...>>::type::ModifierType;
			typename ModifierType::OneShotTrainingParams modifierParams;
			ModifierType::UnpackParameters(modifierParams, params, optimiseParamsMap, mapOffset, paramsOffset);
			auto expandedTuple = std::tuple_cat(modifierTrainingParams, std::make_tuple(modifierParams));
			return CreateModifiersAndCrossValidate<RegressionType, I + 1, ModifierFindMinGlobalTrainingTypes...>(expandedTuple, params,
				optimiseParamsMap,
				mapOffset,
				paramsOffset,
				inputExamples,
				targetExamples,
				randomSeed,
				regressionParams,
				numFolds,
				metric,
				diagnostics);
		}
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
	static Regressor<typename RegressionType::T> RegressorTrainer::TrainRegressorCrossValidation(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
		std::vector<typename RegressionType::T> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		unsigned long const numThreads,
		std::vector<typename RegressionType::T>& diagnostics,
		typename RegressionType::CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
		ModifierCrossValidationTrainingTypes const&... modifiersCrossValidationTrainingPack)
	{
		typedef typename RegressionType::T T;
		static_assert(std::is_base_of<Regressors::RegressionTypes::RegressionComponentBase, RegressionType>::value);
		DLIB_ASSERT(dlib::is_learning_problem(inputExamples, targetExamples),
			"Bad input data.");

		std::tuple<ModifierCrossValidationTrainingTypes...> modifierCrossValidationParams(modifiersCrossValidationTrainingPack...);
		std::vector<typename RegressionType::OneShotTrainingParams> regressionParamsToTry;
		auto const numRegressionParamSetsToTry = RegressionType::IterateRegressionParams(regressionCrossValidationTrainingParams, regressionParamsToTry);
		auto const numModifierParamSetsToTry = NumModifiersCrossValidationPermutations(modifierCrossValidationParams);

		dlib::thread_pool tp(numThreads);
		std::vector<std::pair<Regressor<T>, std::vector<T>>> allCrossValidatedRegressors(numRegressionParamSetsToTry * numModifierParamSetsToTry);
		ProcessModifierCrossValidationParameterPack<RegressionType>(inputExamples, targetExamples, randomSeed, regressionParamsToTry, numFolds, metric, allCrossValidatedRegressors, tp, modifierCrossValidationParams);
		size_t bestIndex = 0;
		for (size_t i = 0; i < allCrossValidatedRegressors.size(); ++i)
		{
			if (allCrossValidatedRegressors[i].first.GetTrainingError() < allCrossValidatedRegressors[bestIndex].first.GetTrainingError())
			{
				bestIndex = i;
			}
		}
		diagnostics = allCrossValidatedRegressors[bestIndex].second;
		return allCrossValidatedRegressors[bestIndex].first;
	}

	template <typename T, size_t TotalNumParams>
	static void RegressorTrainer::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap,
		unsigned const offset)
	{
	}

	template <typename T, size_t TotalNumParams, class... ModifierFindMinGlobalTrainingTypes, class ModifierFindMinGlobalTrainingType>
	static void RegressorTrainer::ConfigureModifierMapping(std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap, 
		unsigned const offset,
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
		unsigned const mapOffset,
		unsigned& paramsOffset)
	{

	}

	template <typename T, size_t TotalNumParams, class ModifierFindMinGlobalTrainingType, class... ModifierFindMinGlobalTrainingTypes>
	static void RegressorTrainer::PackageModifierParams(col_vector<T>& lowerBound,
		col_vector<T>& upperBound,
		std::vector<bool>& isIntegerParam,
		std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
		unsigned const mapOffset,
		unsigned& paramsOffset,
		ModifierFindMinGlobalTrainingType const& first,
		ModifierFindMinGlobalTrainingTypes const&... rest)
	{
		ModifierFindMinGlobalTrainingType::ModifierType::PackageParameters(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, mapOffset, paramsOffset, first);
		RegressorTrainer::PackageModifierParams<T>(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, mapOffset + ModifierFindMinGlobalTrainingType::ModifierType::NumModifierParams, paramsOffset, rest...);
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
	static Regressor<typename RegressionType::T> RegressorTrainer::TrainRegressorFindMinGlobal(const std::vector<col_vector<typename RegressionType::T>>& inputExamples,
		std::vector<typename RegressionType::T> const& targetExamples,
		std::string const& randomSeed,
		CrossValidationMetric const metric,
		size_t const numFolds,
		typename RegressionType::T const& optimisationTolerance,
		unsigned long const numThreads,
		size_t const maxNumCalls,
		std::vector<typename RegressionType::T>& diagnostics,
		typename RegressionType::FindMinGlobalTrainingParams const& regressionFindMinGlobalTrainingParams,
		ModifierFindMinGlobalTrainingTypes const&... modifiersFindMinGlobalTrainingPack)
	{
		typedef typename RegressionType::T T;
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

		unsigned paramsOffset = 0u;
		RegressionType::PackageParameters(lowerBound, upperBound, isIntegerParam, regressionFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		PackageModifierParams<T>(lowerBound, upperBound, isIntegerParam, optimiseParamsMap, RegressionType::NumTotalParams, paramsOffset, modifiersFindMinGlobalTrainingPack...);

		dlib::thread_pool tp(numThreads);
		dlib::max_function_calls numCalls(maxNumCalls);
		auto findMinGlobalMetric = [&](col_vector<T> const& params)
		{
			unsigned paramsOffset = 0u;
			typename RegressionType::OneShotTrainingParams regressionParams(params, optimiseParamsMap, paramsOffset);
			std::tuple<> modifierTrainingParams;
			std::vector<T> diagnostics;
			return CreateModifiersAndCrossValidate<RegressionType, 0ull, ModifierFindMinGlobalTrainingTypes...>(modifierTrainingParams,
				params,
				optimiseParamsMap,
				RegressionType::NumTotalParams,
				paramsOffset,
				inputExamples,
				targetExamples,
				randomSeed,
				regressionParams,
				numFolds,
				metric,
				diagnostics).GetTrainingError();
		};

		auto const result = dlib::find_min_global(tp, findMinGlobalMetric, lowerBound, upperBound, isIntegerParam, numCalls, optimisationTolerance);
		paramsOffset = 0u;
		typename RegressionType::OneShotTrainingParams optimisedRegressionParams(result.x, optimiseParamsMap, paramsOffset);
		std::tuple<> modifierFunctions;
		return CreateModifiersAndCrossValidate<RegressionType, 0, ModifierFindMinGlobalTrainingTypes...>(modifierFunctions,
			result.x,
			optimiseParamsMap,
			RegressionType::NumTotalParams,
			paramsOffset, 
			inputExamples, 
			targetExamples,
			randomSeed,
			optimisedRegressionParams,
			numFolds,
			metric,
			diagnostics);
	}

	template <typename T>
	Regressor<T>::Regressor()
	{
		static_assert(std::is_floating_point<T>());
	}

	template <typename T>
	Regressor<T>::~Regressor()
	{
		m_impl.release();
	}

	template <typename T>
	Regressor<T>::Regressor(Regressor const& other)
	{
		this->operator=(other);
	}

	template <typename T>
	Regressor<T>& Regressor<T>::operator=(Regressor const& other)
	{
		if (&other != this)
		{
			if (!other.m_impl)
			{
				m_impl.reset();
			}
			else
			{
				m_impl = std::unique_ptr<impl_base>(other.m_impl.get());
			}
		}
		return *this;
	}

	template <typename T>
	Regressor<T>::Regressor(Regressor&& other) noexcept = default;

	template <typename T>
	Regressor<T>& Regressor<T>::operator=(Regressor&& other) noexcept = default;

	template <typename T>
	T Regressor<T>::Predict(col_vector<T> const& input) const
	{
		return m_impl->Predict(input);
	}

	template <typename T>
	RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& Regressor<T>::GetTrainedRegressorParams() const
	{
		return m_impl->GetTrainedRegressorParams();
	}

	template <typename T>
	constexpr size_t Regressor<T>::GetNumModifiers() const
	{
		return m_impl->GetNumModifiers();
	}

	template <typename T>
	ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& Regressor<T>::GetTrainedModifierParams(size_t const index) const
	{
		return m_impl->GetTrainedModifierParams(index);
	}

	template <typename T>
	T const& Regressor<T>::GetTrainingError() const
	{
		return m_impl->GetTrainingError();
	}

	template <typename T>
	RegressionTypes::RegressorTypes const& Regressor<T>::GetRegresorType() const
	{
		return m_impl->GetRegressorType();
	}

	template <typename T>
	void serialize(Regressor<T> const& item, std::ostream& out)
	{
		bool isNonEmpty = item.m_impl != nullptr;
		dlib::serialize(isNonEmpty, out);
		if (isNonEmpty)
		{
			serialize(item.m_impl->GetRegressorType(), out);
			item.m_impl->Serialize(out);
		}
	}

	template <class RegressionType, size_t I, class... ModifierFunctionTypes>
	void DeserializeModifiers(unsigned const numModifiers,
		std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		typename RegressionType::OneShotTrainingParams const& regressionParams,
		typename RegressionType::DecisionFunction const& function,
		typename RegressionType::T const& trainingError,
		Regressor<typename RegressionType::T>& item,
		std::istream& in)
	{
		typedef typename RegressionType::T T;
		if constexpr (I == Regressor<T>::MaxNumModifiers)
		{
			throw RegressorTrainer::RegressorError("Modifier overflow.");
		}
		else if (I == numModifiers)
		{
			using RegressorType = typename Regressor<T>::template impl<RegressionType, ModifierFunctionTypes...>;
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
				typename ModifierTypes::NormaliserModifier<T>::ModifierFunction modifierFunction;
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
				typename ModifierTypes::InputPCAModifier<T>::ModifierFunction modifierFunction;
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
				typename ModifierTypes::FeatureSelectionModifier<T>::ModifierFunction modifierFunction;
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

	template <class RegressionType, typename T>
	void DelegateDeserialize(Regressor<T>& item, std::istream& in)
	{
		static_assert(std::is_base_of<RegressionTypes::RegressionComponentBase, RegressionType>::value);
		typename RegressionType::OneShotTrainingParams params;
		deserialize(params, in);
		typename RegressionType::DecisionFunction function;
		deserialize(function, in);
		T trainingError(std::numeric_limits<T>::max());
		dlib::deserialize(trainingError, in);
		unsigned numModifiers = 0u;
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

	template <typename T>
	void deserialize(Regressor<T>& item, std::istream& in)
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
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::LinearKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::PolynomialKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::PolynomialKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::RadialBasisKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::RadialBasisKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::SigmoidKernelRidgeRegression:
			{
				typedef RegressionTypes::KernelRidgeRegression<KernelTypes::SigmoidKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::LinearSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::LinearKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::PolynomialSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::PolynomialKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::RadialBasisSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::RadialBasisKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::SigmoidSupportVectorRegression:
			{
				typedef RegressionTypes::SupportVectorRegression<KernelTypes::SigmoidKernel<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			case RegressionTypes::RegressorTypes::DenseRandomForestRegression:
			{
				typedef RegressionTypes::RandomForestRegression<KernelTypes::DenseExtractor<T>> RegressorType;
				DelegateDeserialize<RegressorType>(item, in);
				break;
			}
			default:
				throw RegressorTrainer::RegressorError("Unrecognised regressor type.");
			}
		}
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionType::T Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::Predict(col_vector<T> input) const
	{
		ApplyModifiers(ModifierFunctions, input);
		return Function(input);
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionTypes::RegressionComponentBase::RegressionOneShotTrainingParamsBase const& Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetTrainedRegressorParams() const
	{
		return TrainedRegressorParams;
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	constexpr size_t Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetNumModifiers() const
	{
		return std::tuple_size<decltype(ModifierFunctions)>::value;
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetTrainedModifierParams(size_t const index) const
	{
		return GetModifierParams(ModifierFunctions, index);
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	typename RegressionType::T const& Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetTrainingError() const
	{
		return TrainingError;
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	RegressionTypes::RegressorTypes const& Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetRegressorType() const
	{
		return RegressionType::RegressorTypeEnum;
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	void Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::Serialize(std::ostream& out) const
	{
		serialize(TrainedRegressorParams, out);
		serialize(Function, out);
		dlib::serialize(TrainingError, out);
		dlib::serialize(std::tuple_size<decltype(ModifierFunctions)>::value, out);
		impl_base::SerializeModifiers(ModifierFunctions, out);
	}

	template <typename T> template <size_t I, class... ModifierFunctionTypes>
	void Regressor<T>::impl_base::SerializeModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions, std::ostream& out)
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

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes>
	Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::impl(typename RegressionType::DecisionFunction const& function,
		std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		typename RegressionType::T const& trainingError,
		typename RegressionType::OneShotTrainingParams const& regressorTrainingParams) :
		TrainingError(trainingError), TrainedRegressorParams(regressorTrainingParams), Function(function), ModifierFunctions(modifierFunctions)
	{
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes> template <size_t I>
	static void Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::ApplyModifiers(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		col_vector<T>& input)
	{
		if constexpr (sizeof...(ModifierFunctionTypes) == I)
		{
			return;
		}
		else
		{
			auto const& function = std::get<I>(modifierFunctions);
			function.Modify(input);
			ApplyModifiers<I + 1>(modifierFunctions, input);
		}
	}

	template <typename T> template <class RegressionType, class... ModifierFunctionTypes> template <size_t I>
	static typename ModifierTypes::ModifierComponentBase::ModifierOneShotTrainingParamsBase const& Regressor<T>::impl<RegressionType, ModifierFunctionTypes...>::GetModifierParams(std::tuple<ModifierFunctionTypes...> const& modifierFunctions,
		size_t const index)
	{
		if constexpr (sizeof... (ModifierFunctionTypes) == 0 || sizeof... (ModifierFunctionTypes) == I)
		{
			throw RegressorTrainer::RegressorError("modifierFunctions tuple access out of bounds.");
		}
		else if (I == index)
		{
			return std::get<I>(modifierFunctions).GetTrainedParams();
		}
		else
		{
			return GetModifierParams<I + 1>(modifierFunctions, index);
		}
	}
}