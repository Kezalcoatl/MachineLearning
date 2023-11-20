#pragma once

namespace PCA
{
	template<typename SampleType>
	PrincipalComponentAnalysis<SampleType> PrincipalComponentAnalysisTrainer<SampleType>::TrainToTargetVariance(std::vector<SampleType> const& data,
		T const targetVariance,
		size_t const maxModes)
	{
		DLIB_ASSERT(data.size() > 0 && data.begin()->size() > 0 && std::all_of(data.begin(), data.end(), [&](SampleType const& col) { return col.size() == data.begin()->size(); }));
		DLIB_ASSERT(targetVariance > 0.0 && targetVariance <= 1.0);
		DLIB_ASSERT(maxModes > 0);

		std::vector<SampleType> eigenvectors;
		SampleType eigenvalues;
		SampleType sampleMeans;

		CoreTraining(data, maxModes, eigenvectors, eigenvalues, sampleMeans);
		TrimModesForVariance(targetVariance, maxModes, eigenvectors, eigenvalues);

		return PrincipalComponentAnalysis<SampleType>(eigenvectors, eigenvalues, sampleMeans);
	}

	template<typename SampleType>
	void PrincipalComponentAnalysisTrainer<SampleType>::CoreTraining(std::vector<SampleType> const& data,
		size_t const maxModes,
		std::vector<SampleType>& eigenvectors,
		SampleType& eigenvalues,
		SampleType& sampleMeans)
	{
		dlib::matrix<T> data_as_mat(data.begin()->size(), data.size());
		for (size_t row = 0; row < data.begin()->size(); ++row)
		{
			for (size_t col = 0; col < data.size(); ++col)
			{
				data_as_mat(row, col) = data[col](row);
			}
			//dlib::set_colm(data_as_mat, col) = data[col];
		}
		dlib::matrix<T> data_as_rows_local(dlib::trans(data_as_mat));
		Regressors::row_vector<T> means = dlib::sum_rows(data_as_rows_local) / data_as_rows_local.nr();
		sampleMeans = SampleType(means.size());
		for (size_t col = 0; col < data_as_rows_local.nc(); ++col)
		{
			dlib::set_colm(data_as_rows_local, col) = dlib::colm(data_as_rows_local, col) - means(col);
			sampleMeans(col) = means(col);
		}

		dlib::matrix<T> y = data_as_rows_local / std::sqrt(data_as_rows_local.nr() - 1);
		dlib::matrix<T> u, w, eigenvectorsUnsorted;

		// despite svd_fast claiming to be randomized, the seed used within it is always the
		// same and it always gives the same results
		dlib::svd_fast(y, u, w, eigenvectorsUnsorted, maxModes);
		dlib::matrix<T> eigenvaluesUnsorted = dlib::pointwise_multiply(w, w);

		// sort according to highest eigenvalues
		std::vector<size_t> eigenvalueIndices(eigenvaluesUnsorted.size());
		std::iota(eigenvalueIndices.begin(), eigenvalueIndices.end(), 0);
		std::sort(eigenvalueIndices.rbegin(), eigenvalueIndices.rend(), [&](size_t lhs, size_t rhs) -> bool { return eigenvaluesUnsorted(lhs) < eigenvaluesUnsorted(rhs); });

		eigenvectors.resize(eigenvectorsUnsorted.nr(), SampleType(eigenvectorsUnsorted.nc()));
		eigenvalues = SampleType(eigenvalueIndices.size());

		for (size_t index = 0; index < eigenvalueIndices.size(); ++index)
		{
			eigenvectors[index] = dlib::colm(eigenvectorsUnsorted, eigenvalueIndices[index]);
			eigenvalues(index) = eigenvaluesUnsorted(eigenvalueIndices[index]);
		}

		if (eigenvalues(0) <= std::numeric_limits<T>::min())
		{
			throw dlib::error("While performing svd pca model training, the first eigenvalue was zero.");
		}
	}

	template<typename SampleType>
	void PrincipalComponentAnalysisTrainer<SampleType>::TrimModesForVariance(T const variance,
		size_t const maxModes,
		std::vector<SampleType>& eigenvectors,
		SampleType& eigenvalues)
	{
		using namespace dlib;
		const T sumEigenvalues = sum(eigenvalues);
		size_t modeCount = 0;
		T runningTotal = 0;
		for (size_t mode = 0; mode < static_cast<size_t>(eigenvalues.size()); ++mode)
		{
			++modeCount;
			runningTotal += eigenvalues(mode);
			if ((runningTotal / sumEigenvalues) >= variance)
			{
				break;
			}
		}
		if (modeCount > maxModes)
		{
			modeCount = maxModes;
		}
		std::vector<SampleType> newEigenvectors(eigenvectors.size(), SampleType(modeCount));
		SampleType newEigenvalues(modeCount);
		for (size_t mode = 0; mode < modeCount; ++mode)
		{
			newEigenvalues(mode) = eigenvalues(mode);
			for (size_t row = 0; row < eigenvectors.size(); ++row)
			{
				newEigenvectors[row](mode) = eigenvectors[row](mode);
			}
		}
		eigenvectors = newEigenvectors;
		eigenvalues = newEigenvalues;
	}

	template<typename SampleType>
	SampleType PrincipalComponentAnalysis<SampleType>::Encode(SampleType const& data, size_t nModes) const
	{
		DLIB_ASSERT(nVariables() > 0);
		DLIB_ASSERT(data.size() == nVariables());
		DLIB_ASSERT(nModes > 0);

		nModes = std::min(nParams(), nModes);

		SampleType params(nModes);

		SampleType normData = data - SampleMeans;

		for (size_t col = 0; col < params.size(); ++col)
		{
			params(col) = 0.0;
			for (size_t row = 0; row < normData.size(); ++row)
			{
				params(col) += Eigenvectors[row](col) * normData(row);
			}
		}

		return params;
	}

	template<typename SampleType>
	SampleType PrincipalComponentAnalysis<SampleType>::Decode(SampleType const& params) const
	{
		DLIB_ASSERT(params.size() <= nParams());

		SampleType result(nVariables);
		for (size_t row = 0; row < result.size(); ++row)
		{
			result(row) = 0.0;
			for (size_t col = 0; col < params.size(); ++col)
			{
				result(row) += Eigenvectors[row](col) * params(col);
			}
			result(row) += SampleMeans(row);
		}
		return result;
	}

	template<typename SampleType>
	size_t PrincipalComponentAnalysis<SampleType>::nParams() const
	{
		return Eigenvalues.size();
	}

	template<typename SampleType>
	size_t PrincipalComponentAnalysis<SampleType>::nVariables() const
	{
		return SampleMeans.size();
	}

	template<typename SampleType>
	void serialize(const PrincipalComponentAnalysis<SampleType>& item, std::ostream& out)
	{
		using namespace dlib;
		serialize(item.Eigenvectors, out);
		serialize(item.Eigenvalues, out);
		serialize(item.SampleMeans, out);
	}

	template<typename SampleType>
	void deserialize(PrincipalComponentAnalysis<SampleType>& item, std::istream& in)
	{
		using namespace dlib;
		deserialize(item.Eigenvectors, in);
		deserialize(item.Eigenvalues, in);
		deserialize(item.SampleMeans, in);
	}
}