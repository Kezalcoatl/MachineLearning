#pragma once
#include <MLLib/TypeDefinitions.h>
#include <vector>
#include <iostream>

namespace PCA
{
	template<typename SampleType>
	class PrincipalComponentAnalysis
	{
		typedef typename SampleType::type T;
		template <typename U>
		friend class PrincipalComponentAnalysisTrainer;
	private:
		PrincipalComponentAnalysis(std::vector<SampleType> const& eigenvectors,
			SampleType const& eigenvalues,
			SampleType const& sampleMeans) :
			Eigenvectors(eigenvectors),
			Eigenvalues(eigenvalues),
			SampleMeans(sampleMeans)
		{}

	public:

		SampleType Encode(SampleType const& data, size_t nModes) const;

		SampleType Decode(SampleType const& params) const;

		size_t nParams() const;

		size_t nVariables() const;

		template<typename SampleType2>
		friend void serialize(PrincipalComponentAnalysis<SampleType2> const& item, std::ostream& out);

		template<typename SampleType2>
		friend void deserialize(PrincipalComponentAnalysis<SampleType2>& item, std::istream& in);

		PrincipalComponentAnalysis()
		{
			static_assert(std::is_floating_point<T>());
		};


	private:
		std::vector<SampleType> Eigenvectors;
		SampleType Eigenvalues;
		SampleType SampleMeans;
	};

	template <typename SampleType>
	class PrincipalComponentAnalysisTrainer
	{
		typedef typename SampleType::type T;
	public:
		PrincipalComponentAnalysisTrainer() = delete;

		static PrincipalComponentAnalysis<SampleType> TrainToTargetVariance(std::vector<SampleType> const& data,
			T const targetVariance,
			size_t const maxModes);

	private:
		static void CoreTraining(std::vector<SampleType> const& data,
			size_t const max_n_modes,
			std::vector<SampleType>& eigenvectors,
			SampleType& eigenvalues,
			SampleType& sampleMeans);

		static void TrimModesForVariance(T const variance,
			size_t const max_n_modes,
			std::vector<SampleType>& eigenvectors,
			SampleType& eigenvalues);

	};
}

#include "impl/PrincipalComponentAnalysis.hpp"