#pragma once

namespace Regressors
{
	namespace RegressionTypes
	{
		template <class KernelType>
		KernelRidgeRegression<KernelType>::OneShotTrainingParams::OneShotTrainingParams() : 
			MaxBasisFunctions(400),
			Lambda(1.e-6)
		{
		}

		template <class KernelType> template <size_t TotalNumParams>
		KernelRidgeRegression<KernelType>::OneShotTrainingParams::OneShotTrainingParams(col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				MaxBasisFunctions = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				MaxBasisFunctions = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[1].first)
			{
				Lambda = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				Lambda = optimiseParamsMap[1].second;
			}
			KernelType::UnpackParameters(KernelOneShotTrainingParams, vecParams, optimiseParamsMap, NumRegressionParams, paramsOffset);
		}

		template <class KernelType>
		ERegressorTypes KernelRidgeRegression<KernelType>::OneShotTrainingParams::GetRegressionType() const
		{
			return KernelRidgeRegression::RegressorTypeEnum;
		}

		template <class KernelType>
		KernelRidgeRegression<KernelType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			MaxBasisFunctionsToTry = { temp.MaxBasisFunctions };
			LambdaToTry = { temp.Lambda };
		}

		template <class KernelType>
		KernelRidgeRegression<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerMaxBasisFunctions = temp.MaxBasisFunctions;
			UpperMaxBasisFunctions = temp.MaxBasisFunctions;
			LowerLambda = temp.Lambda;
			UpperLambda = temp.Lambda;
		}

		template <class KernelType> template <class... ModifierFunctionTypes>
		static impl<KernelRidgeRegression<KernelType>, ModifierFunctionTypes...> KernelRidgeRegression<KernelType>::Train(std::vector<SampleType> const& inputExamples,
			std::vector<T> const& targetExamples,
			OneShotTrainingParams const& regressionTrainingParams,
			std::vector<T>& LeaveOneOutValues,
			T const& trainingError,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions)
		{
			dlib::krr_trainer<typename KernelType::KernelFunctionType> finalTrainer;
			finalTrainer.set_kernel(KernelType::GetKernel(regressionTrainingParams.KernelOneShotTrainingParams));
			finalTrainer.set_max_basis_size(regressionTrainingParams.MaxBasisFunctions);
			finalTrainer.set_lambda(regressionTrainingParams.Lambda);
			return impl<KernelRidgeRegression<KernelType>, ModifierFunctionTypes...>(finalTrainer.train(inputExamples, targetExamples, LeaveOneOutValues), modifierFunctions, trainingError, regressionTrainingParams);
		}

		template <class KernelType>
		void KernelRidgeRegression<KernelType>::IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			std::vector<OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(regressionCrossValidationTrainingParams.MaxBasisFunctionsToTry.size() > 0 && regressionCrossValidationTrainingParams.LambdaToTry.size() > 0,
				"Every regression parameter must have at least one value to try for cross-validation.");

			OneShotTrainingParams osTrainingParams;
			for (const auto& mbf : regressionCrossValidationTrainingParams.MaxBasisFunctionsToTry)
			{
				osTrainingParams.MaxBasisFunctions = mbf;
				for (const auto& l : regressionCrossValidationTrainingParams.LambdaToTry)
				{
					osTrainingParams.Lambda = l;
					KernelType::template IterateKernelParams<KernelRidgeRegression>(osTrainingParams, regressionCrossValidationTrainingParams.KernelCrossValidationTrainingParams, regressionParamSets);
				}
			}
		}

		template <class KernelType> template <size_t TotalNumParams>
		static void KernelRidgeRegression<KernelType>::PackageParameters(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerMaxBasisFunctions;
				upperBound(paramsOffset) = fmgTrainingParams.UpperMaxBasisFunctions;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerLambda;
				upperBound(paramsOffset) = fmgTrainingParams.UpperLambda;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			KernelType::PackageParameters(NumRegressionParams, lowerBound, upperBound, isIntegerParam, fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		}

		template <class KernelType> template <size_t TotalNumParams>
		static void KernelRidgeRegression<KernelType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap)
		{
			optimiseParamsMap[0].first = fmgTrainingParams.LowerMaxBasisFunctions != fmgTrainingParams.UpperMaxBasisFunctions;
			optimiseParamsMap[0].second = fmgTrainingParams.LowerMaxBasisFunctions;
			optimiseParamsMap[1].first = fmgTrainingParams.LowerLambda != fmgTrainingParams.UpperLambda;
			optimiseParamsMap[1].second = fmgTrainingParams.LowerLambda;
			KernelType::ConfigureMapping(fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, NumRegressionParams);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class KernelType>
		SupportVectorRegression<KernelType>::OneShotTrainingParams::OneShotTrainingParams() :
			C(1.0),
			Epsilon(1.e-3),
			EpsilonInsensitivity(0.1),
			CacheSize(200)
		{
		}

		template <class KernelType> template <size_t TotalNumParams>
		SupportVectorRegression<KernelType>::OneShotTrainingParams::OneShotTrainingParams(col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				C = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				C = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[1].first)
			{
				Epsilon = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				Epsilon = optimiseParamsMap[1].second;
			}
			if (optimiseParamsMap[2].first)
			{
				EpsilonInsensitivity = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				EpsilonInsensitivity = optimiseParamsMap[2].second;
			}
			if (optimiseParamsMap[3].first)
			{
				CacheSize = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				CacheSize = optimiseParamsMap[3].second;
			}
			KernelType::UnpackParameters(KernelOneShotTrainingParams, vecParams, optimiseParamsMap, NumRegressionParams, paramsOffset);
		}

		template <class KernelType>
		ERegressorTypes SupportVectorRegression<KernelType>::OneShotTrainingParams::GetRegressionType() const
		{
			return SupportVectorRegression::RegressorTypeEnum;
		}

		template <class KernelType>
		SupportVectorRegression<KernelType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			CToTry = { temp.C };
			EpsilonToTry = { temp.Epsilon };
			EpsilonInsensitivityToTry = { temp.EpsilonInsensitivity };
			CacheSizeToTry = { temp.CacheSize };
		}

		template <class KernelType>
		SupportVectorRegression<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerC = temp.C;
			UpperC = temp.C;
			LowerEpsilon = temp.Epsilon;
			UpperEpsilon = temp.Epsilon;
			LowerEpsilonInsensitivity = temp.EpsilonInsensitivity;
			UpperEpsilonInsensitivity = temp.EpsilonInsensitivity;
			LowerCacheSize = temp.CacheSize;
			UpperCacheSize = temp.CacheSize;
		}

		template <class KernelType> template <class... ModifierFunctionTypes>
		static impl<SupportVectorRegression<KernelType>, ModifierFunctionTypes...> SupportVectorRegression<KernelType>::Train(std::vector<SampleType> const& inputExamples,
			std::vector<T> const& targetExamples,
			OneShotTrainingParams const& regressionTrainingParams,
			std::vector<T>& Residuals,
			T const& trainingError,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions)
		{
			dlib::svr_trainer<typename KernelType::KernelFunctionType> finalTrainer;
			finalTrainer.set_kernel(KernelType::GetKernel(regressionTrainingParams.KernelOneShotTrainingParams));
			finalTrainer.set_c(regressionTrainingParams.C);
			finalTrainer.set_epsilon(regressionTrainingParams.Epsilon);
			finalTrainer.set_epsilon_insensitivity(regressionTrainingParams.EpsilonInsensitivity);
			finalTrainer.set_cache_size(regressionTrainingParams.CacheSize);
			DecisionFunction const df = finalTrainer.train(inputExamples, targetExamples);
			Residuals.resize(targetExamples.size());
			for (size_t i = 0; i < targetExamples.size(); ++i)
			{
				Residuals[i] = df(inputExamples[i]) - targetExamples[i];
			}
			return impl<SupportVectorRegression<KernelType>, ModifierFunctionTypes...>(df, modifierFunctions, trainingError, regressionTrainingParams);
		}

		template <class KernelType>
		void SupportVectorRegression<KernelType>::IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			std::vector<OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(regressionCrossValidationTrainingParams.CToTry.size() > 0 && regressionCrossValidationTrainingParams.EpsilonToTry.size() > 0 && regressionCrossValidationTrainingParams.EpsilonInsensitivityToTry.size() > 0 && regressionCrossValidationTrainingParams.CacheSizeToTry.size() > 0,
				"For cross-validation, every regression parameter must have at least one value to try.");

			OneShotTrainingParams osTrainingParams;
			for (const auto& c : regressionCrossValidationTrainingParams.CToTry)
			{
				osTrainingParams.C = c;
				for (const auto& e : regressionCrossValidationTrainingParams.EpsilonToTry)
				{
					osTrainingParams.Epsilon = e;
					for (const auto& ei : regressionCrossValidationTrainingParams.EpsilonInsensitivityToTry)
					{
						osTrainingParams.EpsilonInsensitivity = ei;
						for (const auto& cs : regressionCrossValidationTrainingParams.CacheSizeToTry)
						{
							osTrainingParams.CacheSize = cs;
							KernelType::template IterateKernelParams<SupportVectorRegression>(osTrainingParams, regressionCrossValidationTrainingParams.KernelCrossValidationTrainingParams, regressionParamSets);
						}
					}
				}
			}
		}

		template <class KernelType> template <size_t TotalNumParams>
		static void SupportVectorRegression<KernelType>::PackageParameters(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerC;
				upperBound(paramsOffset) = fmgTrainingParams.UpperC;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerEpsilon;
				upperBound(paramsOffset) = fmgTrainingParams.UpperEpsilon;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[2].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerEpsilonInsensitivity;
				upperBound(paramsOffset) = fmgTrainingParams.UpperEpsilonInsensitivity;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[3].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerCacheSize;
				upperBound(paramsOffset) = fmgTrainingParams.UpperCacheSize;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			KernelType::PackageParameters(NumRegressionParams, lowerBound, upperBound, isIntegerParam, fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		}

		template <class KernelType> template <size_t TotalNumParams>
		static void SupportVectorRegression<KernelType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap)
		{
			optimiseParamsMap[0].first = fmgTrainingParams.LowerC != fmgTrainingParams.UpperC;
			optimiseParamsMap[0].second = fmgTrainingParams.LowerC;
			optimiseParamsMap[1].first = fmgTrainingParams.LowerEpsilon != fmgTrainingParams.UpperEpsilon;
			optimiseParamsMap[1].second = fmgTrainingParams.LowerEpsilon;
			optimiseParamsMap[2].first = fmgTrainingParams.LowerEpsilonInsensitivity != fmgTrainingParams.UpperEpsilonInsensitivity;
			optimiseParamsMap[2].second = fmgTrainingParams.LowerEpsilonInsensitivity;
			optimiseParamsMap[3].first = fmgTrainingParams.LowerCacheSize != fmgTrainingParams.UpperCacheSize;
			optimiseParamsMap[3].second = fmgTrainingParams.LowerCacheSize;
			KernelType::ConfigureMapping(fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, NumRegressionParams);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class ExtractorType>
		RandomForestRegression<ExtractorType>::OneShotTrainingParams::OneShotTrainingParams() :
			NumTrees(1000),
			MinSamplesPerLeaf(5),
			SubsamplingFraction(1.0/3.0)
		{
		}

		template <class ExtractorType> template <size_t TotalNumParams>
		RandomForestRegression<ExtractorType>::OneShotTrainingParams::OneShotTrainingParams(col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				NumTrees = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				NumTrees = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[1].first)
			{
				MinSamplesPerLeaf = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				MinSamplesPerLeaf = optimiseParamsMap[1].second;
			}
			if (optimiseParamsMap[2].first)
			{
				SubsamplingFraction = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				SubsamplingFraction = optimiseParamsMap[2].second;
			}
			ExtractorType::UnpackParameters(ExtractorOneShotTrainingParams, vecParams, optimiseParamsMap, NumRegressionParams, paramsOffset);
		}

		template <class ExtractorType>
		ERegressorTypes RandomForestRegression<ExtractorType>::OneShotTrainingParams::GetRegressionType() const
		{
			return RandomForestRegression::RegressorTypeEnum;
		}

		template <class ExtractorType>
		RandomForestRegression<ExtractorType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			NumTreesToTry = { temp.NumTrees };
			MinSamplesPerLeafToTry = { temp.MinSamplesPerLeaf };
			SubsamplingFractionToTry = { temp.SubsamplingFraction };
		}

		template <class KernelType>
		RandomForestRegression<KernelType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerNumTrees = temp.NumTrees;
			UpperNumTrees = temp.NumTrees;
			LowerMinSamplesPerLeaf = temp.MinSamplesPerLeaf;
			UpperMinSamplesPerLeaf = temp.MinSamplesPerLeaf;
			LowerSubsamplingFraction = temp.SubsamplingFraction;
			UpperSubsamplingFraction = temp.SubsamplingFraction;
		}

		template <class ExtractorType> template <class... ModifierFunctionTypes>
		static impl<RandomForestRegression<ExtractorType>, ModifierFunctionTypes...> RandomForestRegression<ExtractorType>::Train(std::vector<SampleType> const& inputExamples,
			std::vector<T> const& targetExamples,
			OneShotTrainingParams const& regressionTrainingParams,
			std::vector<T>& OutOfBagValues,
			T const& trainingError,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions)
		{
			dlib::random_forest_regression_trainer<typename ExtractorType::ExtractorFunctionType> finalTrainer;
			finalTrainer.set_feature_extractor(ExtractorType::GetExtractor(regressionTrainingParams.ExtractorOneShotTrainingParams));
			finalTrainer.set_num_trees(regressionTrainingParams.NumTrees);
			finalTrainer.set_min_samples_per_leaf(regressionTrainingParams.MinSamplesPerLeaf);
			finalTrainer.set_feature_subsampling_fraction(regressionTrainingParams.SubsamplingFraction);
			return impl<RandomForestRegression<ExtractorType>, ModifierFunctionTypes...>(finalTrainer.train(inputExamples, targetExamples, OutOfBagValues), modifierFunctions, trainingError, regressionTrainingParams);
		}

		template <class ExtractorType>
		void RandomForestRegression<ExtractorType>::IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			std::vector<OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(regressionCrossValidationTrainingParams.NumTreesToTry.size() > 0 && regressionCrossValidationTrainingParams.MinSamplesPerLeafToTry.size() > 0 && regressionCrossValidationTrainingParams.SubsamplingFractionToTry.size() > 0,
				"For cross-validation, every regression parameter must have at least one value to try.");

			OneShotTrainingParams osTrainingParams;
			for (const auto& nt : regressionCrossValidationTrainingParams.NumTreesToTry)
			{
				osTrainingParams.NumTrees = nt;
				for (const auto& mspl : regressionCrossValidationTrainingParams.MinSamplesPerLeafToTry)
				{
					osTrainingParams.MinSamplesPerLeaf = mspl;
					for (const auto& sf : regressionCrossValidationTrainingParams.SubsamplingFractionToTry)
					{
						osTrainingParams.SubsamplingFraction = sf;
						ExtractorType::template IterateExtractorParams<RandomForestRegression>(osTrainingParams, regressionCrossValidationTrainingParams.ExtractorCrossValidationTrainingParams, regressionParamSets);
					}
				}
			}
		}

		template <class ExtractorType> template <size_t TotalNumParams>
		static void RandomForestRegression<ExtractorType>::PackageParameters(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerNumTrees;
				upperBound(paramsOffset) = fmgTrainingParams.UpperNumTrees;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerMinSamplesPerLeaf;
				upperBound(paramsOffset) = fmgTrainingParams.UpperMinSamplesPerLeaf;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[2].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerSubsamplingFraction;
				upperBound(paramsOffset) = fmgTrainingParams.UpperSubsamplingFraction;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			ExtractorType::PackageParameters(NumRegressionParams, lowerBound, upperBound, isIntegerParam, fmgTrainingParams.ExtractorFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		}

		template <class ExtractorType> template <size_t TotalNumParams>
		static void RandomForestRegression<ExtractorType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap)
		{
			optimiseParamsMap[0].first = fmgTrainingParams.LowerNumTrees != fmgTrainingParams.UpperNumTrees;
			optimiseParamsMap[0].second = fmgTrainingParams.LowerNumTrees;
			optimiseParamsMap[1].first = fmgTrainingParams.LowerMinSamplesPerLeaf != fmgTrainingParams.UpperMinSamplesPerLeaf;
			optimiseParamsMap[1].second = fmgTrainingParams.LowerMinSamplesPerLeaf;
			optimiseParamsMap[2].first = fmgTrainingParams.LowerSubsamplingFraction != fmgTrainingParams.UpperSubsamplingFraction;
			optimiseParamsMap[2].second = fmgTrainingParams.LowerSubsamplingFraction;
			ExtractorType::ConfigureMapping(fmgTrainingParams.ExtractorFindMinGlobalTrainingParams, optimiseParamsMap, NumRegressionParams);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		template <class LinkFunctionType>
		IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::OneShotTrainingParams::OneShotTrainingParams() :
			MaxNumIterations(100),
			ConvergenceTolerance(1.e-3),
			MaxBasisFunctions(400),
			Lambda(1.e-6)
		{
		}

		template <class LinkFunctionType> template <size_t TotalNumParams>
		IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::OneShotTrainingParams::OneShotTrainingParams(col_vector<T> const& vecParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				MaxNumIterations = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				MaxNumIterations = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[1].first)
			{
				ConvergenceTolerance = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				ConvergenceTolerance = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[2].first)
			{
				MaxBasisFunctions = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				MaxBasisFunctions = optimiseParamsMap[0].second;
			}
			if (optimiseParamsMap[3].first)
			{
				Lambda = vecParams(paramsOffset);
				++paramsOffset;
			}
			else
			{
				Lambda = optimiseParamsMap[1].second;
			}
			LinkFunctionType::UnpackParameters(LinkFunctionOneShotTrainingParams, vecParams, optimiseParamsMap, NumRegressionParams, paramsOffset);
			KernelType::UnpackParameters(KernelOneShotTrainingParams, vecParams, optimiseParamsMap, NumRegressionParams + LinkFunctionType::NumLinkFunctionParams, paramsOffset);
		}

		template <class LinkFunctionType>
		ERegressorTypes IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::OneShotTrainingParams::GetRegressionType() const
		{
			return IterativelyReweightedLeastSquaresRegression::RegressorTypeEnum;
		}

		template <class LinkFunctionType>
		IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::CrossValidationTrainingParams::CrossValidationTrainingParams()
		{
			OneShotTrainingParams temp;
			MaxNumIterationsToTry = { temp.MaxNumIterations };
			ConvergenceToleranceToTry = { temp.ConvergenceTolerance };
			MaxBasisFunctionsToTry = { temp.MaxBasisFunctions };
			LambdaToTry = { temp.Lambda };
		}

		template <class LinkFunctionType>
		IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::FindMinGlobalTrainingParams::FindMinGlobalTrainingParams()
		{
			OneShotTrainingParams temp;
			LowerMaxNumIterations = temp.MaxNumIterations;
			UpperMaxNumIterations = temp.MaxNumIterations;
			LowerConvergenceTolerance = temp.ConvergenceTolerance;
			UpperConvergenceTolerance = temp.ConvergenceTolerance;
			LowerMaxBasisFunctions = temp.MaxBasisFunctions;
			UpperMaxBasisFunctions = temp.MaxBasisFunctions;
			LowerLambda = temp.Lambda;
			UpperLambda = temp.Lambda;
		}

		template <class LinkFunctionType> template <class... ModifierFunctionTypes>
		static impl<IterativelyReweightedLeastSquaresRegression<LinkFunctionType>, ModifierFunctionTypes...> IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::Train(std::vector<SampleType> const& inputExamples,
			std::vector<T> const& targetExamples,
			OneShotTrainingParams const& regressionTrainingParams,
			std::vector<T>& Residuals,
			T const& trainingError,
			std::tuple<ModifierFunctionTypes...> const& modifierFunctions)
		{
			GKMTrainer<LinkFunctionType> finalTrainer;
			finalTrainer.SetLinkFunctionParameters(regressionTrainingParams.LinkFunctionOneShotTrainingParams);
			finalTrainer.SetKernel(LinkFunctionType::KernelType::GetKernel(regressionTrainingParams.KernelOneShotTrainingParams));
			finalTrainer.SetMaxBasisFunctions(regressionTrainingParams.MaxBasisFunctions);
			finalTrainer.SetLambda(regressionTrainingParams.Lambda);
			finalTrainer.SetMaxNumIterations(regressionTrainingParams.MaxNumIterations);
			finalTrainer.SetConvergenceTolerance(regressionTrainingParams.ConvergenceTolerance);
			DecisionFunction const df = finalTrainer.Train(inputExamples, targetExamples);
			Residuals.resize(targetExamples.size());
			for (size_t i = 0; i < targetExamples.size(); ++i)
			{
				Residuals[i] = df(inputExamples[i]) - targetExamples[i];
			}
			return impl<IterativelyReweightedLeastSquaresRegression<LinkFunctionType>, ModifierFunctionTypes...>(df, modifierFunctions, trainingError, regressionTrainingParams);
		}

		template <class LinkFunctionType>
		void IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::IterateRegressionParams(CrossValidationTrainingParams const& regressionCrossValidationTrainingParams,
			std::vector<OneShotTrainingParams>& regressionParamSets)
		{
			DLIB_ASSERT(regressionCrossValidationTrainingParams.MaxNumIterationsToTry.size() > 0
				&& regressionCrossValidationTrainingParams.ConvergenceToleranceToTry.size() > 0
				&& regressionCrossValidationTrainingParams.MaxBasisFunctionsToTry.size() > 0
				&& regressionCrossValidationTrainingParams.LambdaToTry.size() > 0,
				"Every regression parameter must have at least one value to try for cross-validation.");

			OneShotTrainingParams osTrainingParams;
			for (const auto& mni : regressionCrossValidationTrainingParams.MaxNumIterationsToTry)
			{
				osTrainingParams.MaxNumIterations = mni;
				for (const auto& ct : regressionCrossValidationTrainingParams.ConvergenceToleranceToTry)
				{
					osTrainingParams.ConvergenceTolerance = ct;
					for (const auto& mbf : regressionCrossValidationTrainingParams.MaxBasisFunctionsToTry)
					{
						osTrainingParams.MaxBasisFunctions = mbf;
						for (const auto& l : regressionCrossValidationTrainingParams.LambdaToTry)
						{
							osTrainingParams.Lambda = l;
							LinkFunctionType::template IterateLinkFunctionParams<IterativelyReweightedLeastSquaresRegression>(osTrainingParams, regressionCrossValidationTrainingParams.LinkFunctionCrossValidationTrainingParams, regressionCrossValidationTrainingParams.KernelCrossValidationTrainingParams, regressionParamSets);
						}
					}
				}
			}
		}

		template <class LinkFunctionType> template <size_t TotalNumParams>
		static void IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::PackageParameters(col_vector<T>& lowerBound,
			col_vector<T>& upperBound,
			std::vector<bool>& isIntegerParam,
			FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams> const& optimiseParamsMap,
			size_t& paramsOffset)
		{
			if (optimiseParamsMap[0].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerMaxNumIterations;
				upperBound(paramsOffset) = fmgTrainingParams.UpperMaxNumIterations;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[1].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerConvergenceTolerance;
				upperBound(paramsOffset) = fmgTrainingParams.UpperConvergenceTolerance;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			if (optimiseParamsMap[2].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerMaxBasisFunctions;
				upperBound(paramsOffset) = fmgTrainingParams.UpperMaxBasisFunctions;
				isIntegerParam[paramsOffset] = true;
				++paramsOffset;
			}
			if (optimiseParamsMap[3].first)
			{
				lowerBound(paramsOffset) = fmgTrainingParams.LowerLambda;
				upperBound(paramsOffset) = fmgTrainingParams.UpperLambda;
				isIntegerParam[paramsOffset] = false;
				++paramsOffset;
			}
			LinkFunctionType::PackageParameters(NumRegressionParams, lowerBound, upperBound, isIntegerParam, fmgTrainingParams.LinkFunctionFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
			KernelType::PackageParameters(NumRegressionParams + LinkFunctionType::NumLinkFunctionParams, lowerBound, upperBound, isIntegerParam, fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, paramsOffset);
		}

		template <class LinkFunctionType> template <size_t TotalNumParams>
		static void IterativelyReweightedLeastSquaresRegression<LinkFunctionType>::ConfigureMapping(FindMinGlobalTrainingParams const& fmgTrainingParams,
			std::array<std::pair<bool, T>, TotalNumParams>& optimiseParamsMap)
		{
			optimiseParamsMap[0].first = fmgTrainingParams.LowerMaxNumIterations != fmgTrainingParams.UpperMaxNumIterations;
			optimiseParamsMap[0].second = fmgTrainingParams.LowerMaxNumIterations;
			optimiseParamsMap[1].first = fmgTrainingParams.LowerConvergenceTolerance != fmgTrainingParams.UpperConvergenceTolerance;
			optimiseParamsMap[1].second = fmgTrainingParams.LowerConvergenceTolerance;
			optimiseParamsMap[2].first = fmgTrainingParams.LowerMaxBasisFunctions != fmgTrainingParams.UpperMaxBasisFunctions;
			optimiseParamsMap[2].second = fmgTrainingParams.LowerMaxBasisFunctions;
			optimiseParamsMap[3].first = fmgTrainingParams.LowerLambda != fmgTrainingParams.UpperLambda;
			optimiseParamsMap[3].second = fmgTrainingParams.LowerLambda;
			LinkFunctionType::ConfigureMapping(fmgTrainingParams.LinkFunctionFindMinGlobalTrainingParams, optimiseParamsMap, NumRegressionParams);
			KernelType::ConfigureMapping(fmgTrainingParams.KernelFindMinGlobalTrainingParams, optimiseParamsMap, NumRegressionParams + LinkFunctionType::NumLinkFunctionParams);
		}
	}
}