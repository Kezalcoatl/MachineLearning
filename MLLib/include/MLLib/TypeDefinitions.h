#pragma once
#include <string>
#include <sstream>
#include <iostream>

#include <dlib/error.h>
#include <dlib/string.h>
#include <dlib/matrix.h>

namespace Regressors
{
	template <typename T>
	using col_vector = dlib::matrix<T, 0, 1>;
	template <typename T>
	using row_vector = dlib::matrix<T, 1, 0>;

	inline std::string TrimEnumString(std::string const& s)
	{
		std::string::const_iterator it = s.begin();
		while (it != s.end() && isspace(static_cast<unsigned char>(*it)))
		{
			++it;
		}
		std::string::const_reverse_iterator rit = s.rbegin();
		while (rit.base() != it && isspace(static_cast<unsigned char>(*rit)))
		{
			++rit;
		}

		return std::string(it, rit.base());
	}

	inline void SplitEnumArgs(const char* args, std::string array[], int maxElements)
	{
		std::stringstream ss(args);
		std::string sub_str;
		int nIndx = 0;
		while (ss.good() && (nIndx < maxElements))
		{
			std::getline(ss, sub_str, ',');

			auto parts = dlib::split(sub_str, "=");

			if (parts.size() != 1)
			{
				nIndx = dlib::string_cast<int>(parts[1]);
			}

			array[nIndx] = TrimEnumString(parts[0]);
			nIndx++;
		}
	}

	/*
	* The following is a MACRO that declares the following functions in addition to the enum declaration itself
	* 
	* std::string dlib::to_string(T enum)
	* Converts the enumeration to a string
	* 
	* std::string dlib::enum_type_as_string(T enum)
	* Given an example of an enumeration value a string of the enumeration type is returned
	* 
	* int dlib::enum_count(T enum)
	* returns the number of elements in an enumeration
	* 
	* serialize(T e, std::ostream&)
	* writes the enumeration value to the stream
	* 
	* deserialize(T& e, std::istream&)
	* reads the enum value from stream
	* 
	* throw_enum_error(T e, std::string msg)
	* utility function for throwing exceptions adding enum information to the message
	*/
#define DECLARE_ENUM(ename, ...)															\
	enum class ename { __VA_ARGS__, MAX_NUMBER_OF_##ename };								\
	inline std::string to_string(ename en)													\
	{																						\
		const auto MAX_NUMBER_OF_##ename = static_cast<int>(ename::MAX_NUMBER_OF_##ename);	\
		static std::string ename##Strings[MAX_NUMBER_OF_##ename];							\
		if (ename##Strings[0].empty())														\
		{																					\
			SplitEnumArgs(#__VA_ARGS__, ename##Strings, MAX_NUMBER_OF_##ename);				\
		}																					\
		auto asInt = static_cast<int>(en);													\
		return ename##Strings[asInt];														\
	}																						\
	inline std::string enum_type_as_string(ename /*e*/)										\
	{																						\
		return	#ename;																	    \
	}																						\
	inline int enum_count(ename /*e*/)														\
	{																						\
		const auto count = static_cast<int>(ename::MAX_NUMBER_OF_##ename);					\
		return count;																		\
	}																						\
	inline void serialize(ename item, std::ostream& out)									\
	{																						\
		const auto val = static_cast<int>(item);											\
		dlib::serialize(val, out);															\
	}																						\
	inline void deserialize(ename& item, std::istream& in)									\
	{																						\
		auto val(0);																		\
		dlib::deserialize(val, in);															\
		item = static_cast<ename>(val);														\
	}																						\
	inline void throw_enum_error(ename item, const std::string & msg)						\
	{																						\
		std::string full_msg = msg + " (" + enum_type_as_string(item) + ")";				\
		throw enum_error(full_msg);															\
	}

	struct enum_error : public dlib::error
	{
		explicit enum_error(const std::string& message) :
			dlib::error(message) {}
	};

	template <typename T>
	T to_enum(const std::string& as_string)
	{
		static_assert(std::is_enum<T>::value, "Class must be an enumerated type.");

		for (auto i = 0; i < enum_count(T()); ++i)
		{
			auto result = static_cast<T>(i);
			auto cur_enum_as_str = to_string(result);
			if (as_string == cur_enum_as_str)
			{
				return result;
			}
		}

		throw_enum_error(T{}, std::string("Could not convert " + as_string + " to enumeration."));
		return T{};
	}
}