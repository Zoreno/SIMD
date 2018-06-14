/*
 * @file SIMD.h
 * @date 2018-04-21
 * @author Joakim Bertils
 * @brief Implements classes abstracting the SIMD data types.
 */

#pragma once

#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <wmmintrin.h>

#include <iostream>

#define ALIGNED(x) __declspec(align(x)) 
#define ALIGNED8 ALIGNED(8)
#define ALIGNED16 ALIGNED(16)
#define ALIGNED32 ALIGNED(32)

const float PI = 3.1415926f;

class x4float;
class x8float;

/*
 * Math Types
 *
 * Vector2f
 * Vector3f
 * Vector4f
 *
 * Vector2d
 * Vector3d
 * Vector4d
 *
 * Vector2i
 * Vector3i
 * Vector4i
 *
 * Matrix2f
 * Matrix3f
 * Matrix4f
 *
 * Matrix2d
 * Matrix3d
 * Matrix4d
 *
 * Color
 * Quaternion
 *
 * SSE Base types:
 *
 * x2float __m64 SSE
 * x4float __m128 SSE - DONE
 * x8float __m256 AVX - DONE
 *
 * x2double __m128d SSE2
 * x4double __m256d AVX
 *
 * x8int8 __m64 MMX
 * x16int8 __m128i SSE2
 * x32int8 __m256i AVX
 *
 * x4int16 __m64 MMX
 * x8int16 __m128i SSE2
 * x16int16 __m256i AVX
 *
 * x2int32 __m64 MMX
 * x4int32 __m128i SSE2
 * x8int32 __m256i AVX
 *
 * x4int64 __m256i AVX
 *
 * SSE Vector Types:
 *
 * x2Vector2f
 * x4Vector2f
 * x8Vector2f
 * x2Vector3f
 * x4Vector3f
 * x8Vector3f
 * x2Vector4f
 * x4Vector4f
 * x8Vector4f
 *
 * x2Vector2d
 * x2vector3d
 * x2Vector4d
 * x4Vector2d
 * x4Vector3d
 * x4Vector4d
 *
 */

 /*
  * TODO:
  *
  * Sin and cosine for x8float just like for x4float.
  *
  */

  /**
   * @brief Represents 4 floats occupying diffenent lanes in SIMD processing.
   */
class x4float
{
public:

	/*
	 * Constructors
	 */

	 /**
	  * @brief Default constructor
	  */
	x4float();

	/**
	 * @brief Initializes each lane with the same value
	 * @param x Value to initialize with.
	 */
	explicit x4float(const float x);

	/**
	 * @brief Initializes each lane with a separate value.
	 * @param a Value for first lane.
	 * @param b Value for second lane.
	 * @param c Value for third lane.
	 * @param d Value for fourth lane.
	 */
	x4float(const float a, const float b, const float c, const float d);

	/**
	 * @brief Initializes each lane with a value extraced from float vector.
	 * @param vec Vector to get values from.
	 */
	explicit x4float(const float *vec);

	/**
	 * @brief Copy constructor
	 * @param other Object to copy from.
	 */
	x4float(const x4float& other);

	/**
	 * @brief Move constructor
	 * @param other Object to move from.
	 */
	explicit x4float(x4float&& other) noexcept;

	/*
	 * Destructor
	 */

	 /**
	  * @brief Destructor
	  */
	virtual ~x4float();

	/*
	 * Assignment operators
	 */

	 /**
	  * @brief Assigns a value to each lane.
	  * @param x Value to assign.
	  * @return Reference to self.
	  */
	x4float& operator=(const float x);

	/**
	 * @brief Assigns a value to each lane extracted from a vector.
	 * @param vec Vector to get values from.
	 * @return Reference to self.
	 */
	x4float& operator=(const float *vec);

	/**
	 * @brief Copy assignment operator.
	 * @param other Object to copy values from.
	 * @return Reference to self.
	 */
	x4float& operator=(const x4float& other);

	/**
	 * @brief Move assignment operator.
	 * @param other Object to move values from.
	 * @return Reference to self.
	 */
	x4float& operator=(x4float&& other) noexcept;

	/*
	 * Arithmetic operators
	 */

	 /**
	  * @brief Performs addition on each lane.
	  * @param other The values to add.
	  * @return Result of the operation.
	  */
	x4float operator+(const x4float& other) const;

	/**
	 * @brief Performs addition on each lane with float constant.
	 * @param rhs First operand.
	 * @param lhs Second operand.
	 * @return Result of the operation.
	 */
	friend x4float operator+(const x4float& lhs, float rhs);

	/**
	 * @brief Performs addition on each lane with float constant.
	 * @param rhs First operand.
	 * @param lhs Second operand.
	 * @return Result of the operation.
	 */
	friend x4float operator+(float lhs, const x4float& rhs);

	/**
	 * @brief Performs subtraction on each lane.
	 * @param other The values to subtract.
	 * @return Result of the operation.
	 */
	x4float operator-(const x4float& other) const;

	/**
	* @brief Performs subtraction on each lane with float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator-(const x4float& lhs, float rhs);

	/**
	* @brief Performs subtraction on each lane with float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator-(float lhs, const x4float& rhs);

	/**
	 * @brief Performs multiplicaton on each lane.
	 * @param other The values to multiply with.
	 * @return Result of the operation.
	 */
	x4float operator*(const x4float& other) const;

	/**
	* @brief Performs multiplication on each lane by float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator*(const x4float& lhs, float rhs);

	/**
	* @brief Performs multiplication on each lane by float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator*(float lhs, const x4float& rhs);

	/**
	 * @brief Performs division on each lane.
	 * @param other The values of the divisior.
	 * @return Result of the operation.
	 */
	x4float operator/(const x4float& other) const;

	/**
	* @brief Performs division on each lane by float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator/(const x4float& lhs, float rhs);

	/**
	* @brief Performs division on each lane by float constant.
	* @param rhs First operand.
	* @param lhs Second operand.
	* @return Result of the operation.
	*/
	friend x4float operator/(float lhs, const x4float& rhs);

	/**
	 * @brief Unary plus operator.
	 * @return R-value with the values of this object.
	 */
	x4float operator+() const;

	/**
	 * @brief Unary minus operator.
	 * @return R-value with the inverted sign values of this object.
	 */
	x4float operator-() const;

	/**
	* @brief Negates the bitfield of each lane.
	* @return R-value with the inverted bitfields of each lane.
	*/
	x4float operator!() const;

	/*
	 * Relational operators
	 */

	 /**
	  * @brief Performs lane-wise equality comparison.
	  * @param other The values to compare with.
	  * @return Each lane contains the boolean value of the operation.
	  */
	x4float operator==(const x4float& other) const;

	/**
	* @brief Performs lane-wise inequality comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator!=(const x4float& other) const;

	/**
	* @brief Performs lane-wise less-than comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator<(const x4float &other) const;

	/**
	* @brief Performs lane-wise less-than-or-equal comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator<=(const x4float &other) const;

	/**
	* @brief Performs lane-wise greater-than comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator>(const x4float &other) const;

	/**
	* @brief Performs lane-wise greater-than-or-equal comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator>=(const x4float &other) const;

	/*
	 * Bitwise Operators
	 */

	 /**
	  * @brief Performs bitwise AND operation on each lane.
	  * @param other The values to AND with.
	  * @return Each lane contains the boolean value of the operation.
	  */
	x4float operator&(const x4float& other) const;

	/**
	* @brief Performs bitwise OR operation on each lane.
	* @param other The values to OR with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator|(const x4float& other) const;

	/**
	* @brief Performs bitwise XOR operation on each lane.
	* @param other The values to XOR with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x4float operator^(const x4float& other) const;

	/*
	 * Compound operators
	 */

	 /**
	  * @brief Adds a value to this value for each lane.
	  * @param other The values to add.
	  * @return Reference to self.
	  */
	x4float& operator+=(const x4float& other);

	/**
	 * @brief Subtracts a value to this value for each lane.
	 * @param other The values to subtract.
	 * @return Reference to self.
	 */
	x4float& operator-=(const x4float& other);

	/**
	* @brief Multiplies a value to this value for each lane.
	* @param other The values to multiply.
	* @return Reference to self.
	*/
	x4float& operator*=(const x4float& other);

	/**
	* @brief Divides this value by a value for each lane.
	* @param other The values of the divisors.
	* @return Reference to self.
	*/
	x4float& operator/=(const x4float& other);

	/**
	* @brief Perform bitwise AND with a value for each lane.
	* @param other The values to AND with.
	* @return Reference to self.
	*/
	x4float& operator&=(const x4float& other);

	/**
	* @brief Perform bitwise OR with a value for each lane.
	* @param other The values to OR with.
	* @return Reference to self.
	*/
	x4float& operator|=(const x4float& other);

	/**
	* @brief Perform bitwise XOR with a value for each lane.
	* @param other The values to XOR with.
	* @return Reference to self.
	*/
	x4float& operator^=(const x4float& other);

	/*
	 * Stream Operators
	 */

	 /**
	  * @brief Prints the value of this object to a stream.
	  * @param stream Stream to print to.
	  * @param value Value to print.
	  * @return Reference to stream.
	  */
	friend std::ostream& operator<<(std::ostream& stream, const x4float& value);

	/**
	 * @brief Reads a value for each lane from stream.
	 * @param stream Stream to read from.
	 * @param value Value to read to.
	 * @return Reference to stream.
	 */
	friend std::istream& operator >> (std::istream& stream, x4float& value);

	/*
	 * Element access
	 */

	 /**
	  * @brief Gets the value of a specified lane.
	  * @param i Lane index
	  * @return Value of lane.
	  */
	const float& operator[](int i) const;

	/**
	* @brief Gets the value of a specified lane.
	* @param i Lane index
	* @return Value of lane.
	*/
	float& operator[](int i);

	/**
	* @brief Gets the value of a specified lane.
	* @param i Lane index.
	* @return Value of lane.
	*/
	float at(int i) const;

	/*
	* Type casts
	*/


	/**
	 * @brief Type cast to SIMD primitive
	 */
	explicit operator union __m128() const;

	/*
	 * Functions
	 */

	 /**
	  * @brief Calculates the maximum for each lane.
	  * @param a First operand.
	  * @param b Second operand.
	  * @return Maximum value of each lane.
	  */
	friend x4float max(const x4float& a, const x4float& b);

	/**
	* @brief Calculates the minimum for each lane.
	* @param a First operand.
	* @param b Second operand.
	* @return Minimum value of each lane.
	*/
	friend x4float min(const x4float& a, const x4float& b);

	/**
	 * @brief Calculates the reciprocal for each lane.
	 * @param a Operand.
	 * @return Reciprocal for each lane.
	 */
	friend x4float rcp(const x4float& a);

	/**
	 * @brief Calculates the square root for each lane.
	 * @param a Operand.
	 * @return Square root of each lane.
	 */
	friend x4float sqrt(const x4float& a);

	/**
	 * @brief Calculates the reciprocal square root for each lane.
	 * @param a Operand.
	 * @return Reciprocal square roor for each lane.
	 */
	friend x4float rsqrt(const x4float& a);

	/**
	 * @brief Adds odd lane indices and subtracts even lane indices.
	 * @param a First operand.
	 * @param b Second operand.
	 * @return Result of the operation.
	 */
	friend x4float addsub(const x4float& a, const x4float& b);

	/**
	 * @brief Rounds the value of each lane up to an integer.
	 * @param a Operand.
	 * @return Result of the operation.
	 */
	friend x4float ceil(const x4float& a);

	/**
	* @brief Rounds the value of each lane down to an integer.
	* @param a Operand.
	* @return Result of the operation.
	*/
	friend x4float floor(const x4float& a);

	/**
	 * @brief Horizontally add adjacent pairs.
	 *
	 * ret[0] = a[1] + a[0]
	 * ret[1] = a[3] + a[2]
	 * ret[2] = b[1] + b[0]
	 * ret[3] = b[3] + b[2]
	 *
	 * @param a First operand.
	 * @param b Second operand.
	 * @return Result of the operation.
	 */
	friend x4float hadd(const x4float& a, const x4float& b);

	/**
	* @brief Horizontally subtract adjacent pairs.
	*
	* ret[0] = a[0] - a[1]
	* ret[1] = a[2] - a[3]
	* ret[2] = b[0] - b[1]
	* ret[3] = b[2] - b[3]
	*
	* @param a First operand.
	* @param b Second operand.
	* @return Result of the operation.
	*/
	friend x4float hsub(const x4float& a, const x4float& b);

	/**
	 * @brief Blends the values from a and b depending on the value of the mask lane.
	 * @param a First operand.
	 * @param b Second operand.
	 * @param mask Mask operand.
	 * @return Result of the operation.
	 */
	friend x4float blendv(const x4float& a, const x4float& b, const x4float& mask);

	/**
	 * @brief Swaps the contents of two objects.
	 * @param a First operand.
	 * @param b Second operand.
	 */
	friend void swap(x4float& a, x4float& b) noexcept;

	/**
	 * @brief Calculates the modulo of each lane.
	 * @param a First operand.
	 * @param b Second operand.
	 * @return Result of the operation.
	 */
	friend x4float mod(const x4float& a, const x4float& b);

	/**
	 * @brief Calculates the absolute value of each lane.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float abs(const x4float& a);

	/**
	 * @brief Calculates the sine of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float sin(const x4float& a);

	/**
	 * @brief Calculates the cosine of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float cos(const x4float& a);

	/**
	 * @brief Calculates the tangent of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float tan(const x4float& a);

	/**
	 * @brief Calculates arctan of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float atan(const x4float& a);

	/**
	 * @brief Calculates arcsin of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float asin(const x4float& a);

	/**
	* @brief Calculates arccos of the input.
	* @param a The operand.
	* @return Result of the operation.
	*/
	friend x4float acos(const x4float& a);

	/**
	 * @brief Calculates the log base 2 of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float log2(const x4float& a);

	/**
	 * @brief Calculates the natural logarithm of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float log(const x4float& a);

	/**
	 * @brief Calculates the log base 10 of the input.
	 * @param a The operand.
	 * @return Result of the operation.
	 */
	friend x4float log10(const x4float& a);

	/**
	 * @brief Calculates the log for an arbitrary base of the input.
	 * @param a The paramater.
	 * @param b The base.
	 * @return Result of the operation.
	 */
	friend x4float logb(const x4float& a, const x4float& b);

	/**
	 * @brief Calculates 2^a for each lane.
	 * @param a The exponent.
	 * @return Result of the operand.
	 */
	friend x4float pow2(const x4float& a);

	/**
	 * @brief Calculates e^a for each lane.
	 * @param a The exponent.
	 * @return Result of the operation.
	 */
	friend x4float exp(const x4float& a);

	/**
	 * @brief Calculates x^p for each lane.
	 * @param x The base.
	 * @param p The exponent.
	 * @return Result of the operation.
	 */
	friend x4float pow(const x4float& x, const x4float& p);

	/**
	 * @brief Calculates the hyperbolic sine of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float sinh(const x4float& x);

	/**
	 * @brief Calculates the hyperbolic cosine of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float cosh(const x4float& x);

	/**
	 * @brief Calculates the hyperbolic tangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float tanh(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic tangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float atanh(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic sine of the input.
	 * @param x The operand.
	 * @return Result of the operand.
	 */
	friend x4float asinh(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic cosine of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float acosh(const x4float &x);

	/**
	 * @brief Calculates the cosecant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float csc(const x4float& x);

	/**
	 * @brief Calculates the secant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float sec(const x4float& x);

	/**
	 * @brief Calculates the cotangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float cot(const x4float& x);

	/**
	 * @brief Calculates the hyperbolic cosecant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float csch(const x4float& x);

	/**
	 * @brief Calculates the hyperbolic secant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float sech(const x4float& x);

	/**
	 * @brief Calculates the hyperbolic cotangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float coth(const x4float& x);

	/**
	 * @brief Calculates the inverse cosecant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float acsc(const x4float& x);

	/**
	 * @brief Calculates the inverse secant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float asec(const x4float& x);

	/**
	 * @brief Calculates the inverse cotangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float acot(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic cosecant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float acsch(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic secant of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float asech(const x4float& x);

	/**
	 * @brief Calculates the inverse hyperbolic cotangent of the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float acoth(const x4float& x);

	/**
	 * @brief Calculates the natural logarithm of the gamma function for the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float lgamma(const x4float& x);

	/**
	 * @brief Calculates the derivative of the natural logartihm of the gamma function for the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float digamma(const x4float& x);

	/**
	 * @brief Calculates the sigmoid function for the input.
	 * @param x The operand.
	 * @return Result of the operation.
	 */
	friend x4float sigmoid(const x4float& x);

	/**
	 * @brief The values of each lane.
	 */
	__m128 _data;
};

inline x4float::x4float()
{
	// Initialize to the default value of 0.
	_data = _mm_setzero_ps();
}

inline x4float::x4float(const float x)
{
	// Initialize all elements to the value of x.
	_data = _mm_set1_ps(x);
}

inline x4float::x4float(
	const float a,
	const float b,
	const float c,
	const float d)
{
	// Inititalize to the given values
	_data = _mm_setr_ps(a, b, c, d);
}

inline x4float::x4float(const float *vec)
{
	// Load all 4 elements from memory
	_data = _mm_loadu_ps(vec);
}

inline x4float::x4float(const x4float& other)
{
	_data = _mm_load_ps(other._data.m128_f32);
}

inline x4float::x4float(x4float &&other) noexcept
{
	swap(*this, other);
}

inline x4float::~x4float()
{
	// Do nothing for now.
}

inline x4float & x4float::operator=(const float x)
{
	_data = _mm_set1_ps(x);

	return *this;
}

inline x4float & x4float::operator=(const float *vec)
{
	_data = _mm_load_ps(vec);

	return *this;
}

inline x4float & x4float::operator=(const x4float &other)
{
	_data = _mm_load_ps(other._data.m128_f32);

	return *this;
}

inline x4float & x4float::operator=(x4float &&other) noexcept
{
	swap(*this, other);

	return *this;
}

inline x4float x4float::operator+(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_add_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator-(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_sub_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator*(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_mul_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator/(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_div_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator+() const
{
	x4float ret;
	const x4float factor{ 1.0f };

	ret._data = _mm_mul_ps(_data, factor._data);

	return ret;
}

inline x4float x4float::operator-() const
{
	x4float ret;
	const x4float factor{ -1.0f };

	ret._data = _mm_mul_ps(_data, factor._data);

	return ret;
}

inline x4float x4float::operator!() const
{
	x4float ret;

	ret._data = _mm_andnot_ps(_data, _data);

	return ret;
}

inline x4float x4float::operator==(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmpeq_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator!=(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmpneq_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator<(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmplt_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator<=(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmple_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator>(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmpgt_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator>=(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_cmpge_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator&(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_and_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator|(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_or_ps(_data, other._data);

	return ret;
}

inline x4float x4float::operator^(const x4float &other) const
{
	x4float ret;

	ret._data = _mm_xor_ps(_data, other._data);

	return ret;
}

inline x4float & x4float::operator+=(const x4float &other)
{
	_data = _mm_add_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator-=(const x4float &other)
{
	_data = _mm_sub_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator*=(const x4float &other)
{
	_data = _mm_mul_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator/=(const x4float &other)
{
	_data = _mm_div_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator&=(const x4float &other)
{
	_data = _mm_and_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator|=(const x4float &other)
{
	_data = _mm_or_ps(_data, other._data);

	return *this;
}

inline x4float & x4float::operator^=(const x4float &other)
{
	_data = _mm_xor_ps(_data, other._data);

	return *this;
}

inline const float& x4float::operator[](int i) const
{
	return _data.m128_f32[i];
}

inline float & x4float::operator[](int i)
{
	return _data.m128_f32[i];
}

inline float x4float::at(int i) const
{
	const float ret = _data.m128_f32[i];

	return ret;
}

inline x4float::operator union __m128() const
{
	return _data;
}

inline x4float operator+(
	const x4float &lhs,
	float rhs)
{
	return lhs + x4float{ rhs };
}

inline x4float operator+(
	float lhs,
	const x4float &rhs)
{
	return x4float{ lhs } +rhs;
}

inline x4float operator-(
	const x4float &lhs,
	float rhs)
{
	return lhs - x4float{ rhs };
}

inline x4float operator-(
	float lhs,
	const x4float &rhs)
{
	return x4float{ lhs } -rhs;
}

inline x4float operator*(
	const x4float &lhs,
	float rhs)
{
	return lhs * x4float{ rhs };
}

inline x4float operator*(
	float lhs,
	const x4float &rhs)
{
	return x4float{ lhs } *rhs;
}

inline x4float operator/(
	const x4float &lhs,
	float rhs)
{
	return lhs / x4float{ rhs };
}

inline x4float operator/(
	float lhs,
	const x4float &rhs)
{
	return x4float{ lhs } / rhs;
}

inline std::ostream & operator<<(std::ostream &stream, const x4float &value)
{
	stream << value[0] << " " << value[1] << " " << value[2] << " " << value[3];

	return stream;
}

inline std::istream & operator >> (std::istream &stream, x4float &value)
{
	float f0;
	float f1;
	float f2;
	float f3;

	stream >> f0 >> f1 >> f2 >> f3;

	value = x4float{ f0, f1, f2, f3 };

	return stream;
}

inline x4float max(const x4float &a, const x4float &b)
{
	x4float ret;

	ret._data = _mm_max_ps(a._data, b._data);

	return ret;
}

inline x4float min(const x4float &a, const x4float &b)
{
	x4float ret;

	ret._data = _mm_min_ps(a._data, b._data);

	return ret;
}

inline x4float rcp(const x4float &a)
{
	x4float ret;

	ret._data = _mm_rcp_ps(a._data);

	return ret;
}

inline x4float sqrt(const x4float &a)
{
	x4float ret;

	ret._data = _mm_sqrt_ps(a._data);

	return ret;
}

inline x4float rsqrt(const x4float &a)
{
	x4float ret;

	ret._data = _mm_rsqrt_ps(a._data);

	return ret;
}

inline x4float addsub(const x4float &a, const x4float &b)
{
	x4float ret;

	ret._data = _mm_addsub_ps(a._data, b._data);

	return ret;
}

inline x4float ceil(const x4float &a)
{
	x4float ret;

	ret._data = _mm_ceil_ps(a._data);

	return ret;
}

inline x4float floor(const x4float &a)
{
	x4float ret;

	ret._data = _mm_floor_ps(a._data);

	return ret;
}

inline x4float hadd(const x4float &a, const x4float &b)
{
	x4float ret;

	ret._data = _mm_hadd_ps(a._data, b._data);

	return ret;
}

inline x4float hsub(const x4float &a, const x4float &b)
{
	x4float ret;

	ret._data = _mm_hsub_ps(a._data, b._data);

	return ret;
}

inline x4float blendv(const x4float &a, const x4float &b, const x4float &mask)
{
	x4float ret;

	ret._data = _mm_blendv_ps(a._data, b._data, mask._data);

	return ret;
}

inline void swap(x4float &a, x4float &b) noexcept
{
	const __m128 temp = _mm_shuffle_ps(a._data, a._data, _MM_SHUFFLE(3, 2, 1, 0));

	a._data = _mm_shuffle_ps(b._data, b._data, _MM_SHUFFLE(3, 2, 1, 0));
	b._data = _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(3, 2, 1, 0));
}

inline x4float mod(const x4float &a, const x4float &b)
{
	const x4float quot = floor(a / b);

	return a - quot * b;
}

inline x4float abs(const x4float &a)
{
	x4float ret;

	// Extract the sign bit
	const __m128 sign = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

	// Flip the sign bit if it was set.
	ret._data = _mm_andnot_ps(sign, a._data);

	return ret;
}

inline x4float sin(const x4float &a)
{
	//https://dtosoftware.wordpress.com/2013/01/07/fast-sin-and-cos-functions/

	const float B = 4.f / PI;
	const float C = -4.f / (PI * PI);
	const float P = 0.255f;

	x4float m_x{ a };
	const x4float m_pi{ PI };
	const x4float m_mpi{ -PI };
	const x4float m_2pi{ PI * 2.f };
	const x4float m_B{ B };
	const x4float m_C{ C };
	const x4float m_P{ P };

	// Make sure operand is within [-2*PI, 2*PI]

	x4float m1 = m_x >= m_pi;
	m1 &= m_2pi;
	m_x -= m1;
	m1 = (m_x <= m_mpi);
	m1 &= m_2pi;
	m_x += m1;

	const x4float m_abs = abs(m_x);

	x4float m_y = (m_abs * m_C + m_B) * m_x;
	m_y += (abs(m_y) * m_y - m_y) * m_P;

	return m_y;
}

inline x4float cos(const x4float &a)
{
	// Make use of the fact that cos(x) = sin(x + PI/2)

	const x4float x = x4float{ PI / 2.f } -a;

	return sin(x);
}

inline x4float tan(const x4float &a)
{
	return sin(a) / cos(a);
}

inline x4float atan(const x4float &a)
{
	// https://dsp.stackexchange.com/questions/20444/books-resources-for-implementing-various-mathematical-functions-in-fixed-point-a/20482#20482

	x4float x = a;

	const x4float gt1_mask = x >= x4float{ 1.f };

	x = blendv(x, rcp(x), gt1_mask);

	const x4float ltn1_mask = x <= x4float{ -1.f };

	x = blendv(x, rcp(x), ltn1_mask);

	const x4float a0{ 1.0f };
	const x4float a1{ 0.33288950512027f };
	const x4float a2{ -0.08467922817644f };
	const x4float a3{ 0.03252232640125f };
	const x4float a4{ -0.00749305860992f };

	const x4float x2 = x*x;

	const x4float f = a0 + (a1 + (a2 + (a3 + a4*x2)*x2)*x2)*x2;

	x4float res = x / f;

	res = blendv(res, x4float{ PI / 2.f } -res, gt1_mask);
	res = blendv(res, x4float{ -PI / 2.f } -res, ltn1_mask);

	return res;
}

inline x4float asin(const x4float &a)
{
	return atan(a / sqrt(x4float{ 1.f } -a*a));
}

inline x4float acos(const x4float &a)
{
	return x4float{ PI / 2.f } -atan(a / sqrt(x4float{ 1.f } -a*a));
}

inline x4float log2(const x4float &a)
{
	const union
	{
		__m128 f;
		__m128i i;
	} vx{ a._data };

	union
	{
		__m128 f;
		__m128i i;
	} mx;

	mx.i = _mm_or_si128(_mm_and_si128(vx.i, _mm_set1_epi32(0x007FFFFF)), _mm_set1_epi32(0x3F000000));
	x4float y;
	y._data = _mm_cvtepi32_ps(vx.i);
	y *= x4float{ 1.1920928955078125e-7f };

	const x4float c_124_22551499{ 124.22551499f };
	const x4float c_1_498030302{ 1.498030302f };
	const x4float c_1_725877999{ 1.72587999f };
	const x4float c_0_3520087068{ 0.3520887068f };

	x4float mxf;
	mxf._data = mx.f;

	return y - c_124_22551499 - c_1_498030302 * mxf - c_1_725877999 / (c_0_3520087068 + mxf);
}

inline x4float log(const x4float &a)
{
	const x4float c_0_69314718{ 0.69314718f };

	return c_0_69314718 * log2(a);
}

inline x4float log10(const x4float &a)
{
	const x4float c_0_3010299956639812{ 0.3010299956639812f };

	return c_0_3010299956639812 * log2(a);
}

inline x4float logb(const x4float &a, const x4float &b)
{
	return log2(a) / log2(b);
}

inline x4float pow2(const x4float &a)
{
	const int round_mode = _MM_GET_ROUNDING_MODE();

	_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
	const x4float ltzero = a < x4float{ 0.f };
	const x4float offset = ltzero & x4float{ 1.f };
	const x4float lt126 = a < x4float{ -126.f };
	const x4float clipp = blendv(a, x4float{ -126.f }, lt126);
	const __m128i w = _mm_cvtps_epi32(clipp._data);

	x4float w_f;
	w_f._data = _mm_cvtepi32_ps(w);

	const x4float z = clipp - w_f + offset;

	const x4float c_121_2740838{ 121.2740575f };
	const x4float c_27_7280233{ 27.7280233f };
	const x4float c_4_84252568{ 4.84252568f };
	const x4float c_1_49012907{ 1.49012907f };

	union
	{
		__m128 f;
		__m128i i;
	} v;

	const x4float c = x4float{ 1 << 23 } *(clipp + c_121_2740838 + c_27_7280233 / (c_4_84252568 - z) - c_1_49012907 * z);
	v.i = _mm_cvtps_epi32(c._data);

	x4float ret;

	ret._data = v.f;

	_MM_SET_ROUNDING_MODE(round_mode);

	return ret;
}

inline x4float exp(const x4float &a)
{
	return pow2(x4float{ 1.442695040f } *a);
}

inline x4float pow(const x4float &x, const x4float &p)
{
	return pow2(p * log2(x));
}

inline x4float sinh(const x4float &x)
{
	return x4float{ 0.5f } *(exp(x) - exp(-x));
}

inline x4float cosh(const x4float &x)
{
	return x4float{ 0.5f } *(exp(x) + exp(-x));
}

inline x4float tanh(const x4float &x)
{
	return x4float{ -1.0f } +x4float{ 2.f } / (x4float{ 1.f } +exp(x4float{ -2.f } *x));
}

inline x4float atanh(const x4float &x)
{
	// TODO: Check precision of this function.
	return log((x4float{ 1.f } +x) / (x4float{ 1.f } -x)) / x4float{ 2.f };
}

inline x4float asinh(const x4float &x)
{
	// TODO: Check precision of this function.
	return log(x4float{ 1.f } +sqrt(x*x + x4float{ 1.f }));
}

inline x4float acosh(const x4float &x)
{
	// TODO: Check precision of this function.
	return log(x4float{ 1.f } +sqrt(x*x - x4float{ 1.f }));
}

inline x4float csc(const x4float &x)
{
	return x4float{ 1.f } / sin(x);
}

inline x4float sec(const x4float &x)
{
	return x4float{ 1.f } / cos(x);
}

inline x4float cot(const x4float &x)
{
	return x4float{ 1.f } / tan(x);
}

inline x4float csch(const x4float &x)
{
	return x4float{ 2.f } / (exp(x) - exp(-x));
}

inline x4float sech(const x4float &x)
{
	return x4float{ 2.f } / (exp(x) + exp(-x));
}

inline x4float coth(const x4float &x)
{
	return (exp(x) + exp(-x)) / (exp(x) - exp(-x));
}

inline x4float acsc(const x4float &x)
{
	return asin(x4float{ 1.f } / x);
}

inline x4float asec(const x4float &x)
{
	return acos(x4float{ 1.f } / x);
}

inline x4float acot(const x4float &x)
{
	return x4float{ PI / 2.f } -atan(x);
}

inline x4float acsch(const x4float &x)
{
	return x4float{ 1.f } / log(x + sqrt(x*x + x4float{ 1.f }));
}

inline x4float asech(const x4float &x)
{
	return x4float{ 1.f } / log(x + sqrt(x*x - x4float{ 1.f }));
}

inline x4float acoth(const x4float &x)
{
	return log((x + x4float{ 1.f }) / (x - x4float{ 1.f })) / x4float{ 2.f };
}

inline x4float lgamma(const x4float &x)
{
	const x4float c_1_0{ 1.f };
	const x4float c_2_0{ 2.f };
	const x4float c_3_0{ 3.f };
	const x4float c_2_081061466{ 2.081061466f };
	const x4float c_0_0833333{ 0.0833333f };
	const x4float c_2_5{ 2.5f };

	const x4float logterm = log(x * (c_1_0 + x) * (c_2_0 + x));
	const x4float xp3 = c_3_0 + x;

	return -c_2_081061466
		- x
		+ c_0_0833333 / xp3
		- logterm
		+ (c_2_5 + x)*log(xp3);
}

inline x4float digamma(const x4float &x)
{
	const x4float twopx = x4float{ 1.f } +x;
	const x4float logterm = log(twopx);

	const x4float c_n_48_0{ -48.f };
	const x4float c_n_157_0{ -157.f };
	const x4float c_n_127_0{ -127.f };
	const x4float c_30_0{ 30.f };
	const x4float c_12_0{ 12.f };
	const x4float c_1_0{ 1.f };

	return (c_n_48_0 + x * (c_n_157_0 + x * (c_n_127_0 - c_30_0 * x))) /
		(c_12_0 * x * (c_1_0 + x) * twopx * twopx)
		+ logterm;
}

inline x4float sigmoid(const x4float &x)
{
	return x4float{ 1.f } / (x4float{ 1.f } +exp(-x));
}

/**
* @brief Represents 8 floats occupying diffenent lanes in SIMD processing.
*/
class x8float
{
public:

	/*
	* Constructor
	*/

	/**
	* @brief Default constructor
	*/
	x8float();

	/**
	* @brief Initializes each lane with the same value
	* @param x Value to initialize with.
	*/
	explicit x8float(
		float x);

	/**
	* @brief Initializes each lane with a separate value.
	* @param a Value for first lane.
	* @param b Value for second lane.
	* @param c Value for third lane.
	* @param d Value for fourth lane.
	* @param e Value for fifth lane.
	* @param f Value for sixth lane.
	* @param g Value for seventh lane.
	* @param h Value for eight lane.
	*/
	x8float(
		const float a,
		const float b,
		const float c,
		const float d,
		const float e,
		const float f,
		const float g,
		const float h);

	/**
	* @brief Initializes each lane with a value extraced from float vector.
	* @param vec Vector to get values from.
	*/
	explicit x8float(const float *vec);

	x8float(const x4float& a, const x4float& b);

	/**
	* @brief Copy constructor
	* @param other Object to copy from.
	*/
	x8float(const x8float& other);

	/**
	* @brief Move constructor
	* @param other Object to move from.
	*/
	explicit x8float(x8float&& other) noexcept;

	/*
	* Destructor
	*/

	/**
	* @brief Destructor
	*/
	virtual ~x8float();

	/*
	* Assignment operators
	*/

	/**
	* @brief Assigns a value to each lane.
	* @param x Value to assign.
	* @return Reference to self.
	*/
	x8float& operator=(const float x);

	/**
	* @brief Assigns a value to each lane extracted from a vector.
	* @param vec Vector to get values from.
	* @return Reference to self.
	*/
	x8float& operator=(const float *vec);

	/**
	* @brief Copy assignment operator.
	* @param other Object to copy values from.
	* @return Reference to self.
	*/
	x8float& operator=(const x8float& other);

	/**
	* @brief Move assignment operator.
	* @param other Object to move values from.
	* @return Reference to self.
	*/
	x8float& operator=(x8float&& other) noexcept;

	/*
	* Arithmetic operators
	*/

	/**
	* @brief Performs addition on each lane.
	* @param other The values to add.
	* @return Result of the operation.
	*/
	x8float operator+(const x8float& other) const;

	/**
	* @brief Performs subtraction on each lane.
	* @param other The values to subtract.
	* @return Result of the operation.
	*/
	x8float operator-(const x8float& other) const;

	/**
	* @brief Performs multiplicaton on each lane.
	* @param other The values to multiply with.
	* @return Result of the operation.
	*/
	x8float operator*(const x8float& other) const;

	/**
	* @brief Performs division on each lane.
	* @param other The values of the divisor.
	* @return Result of the operation.
	*/
	x8float operator/(const x8float& other) const;

	/**
	* @brief Unary plus operator.
	* @return R-value with the values of this object.
	*/
	x8float operator+(void) const;

	/**
	* @brief Unary minus operator.
	* @return R-value with the inverted sign values of this object.
	*/
	x8float operator-(void) const;

	/*
	* Relational operators
	*/

	/**
	* @brief Performs lane-wise equality comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator==(const x8float& other) const;

	/**
	* @brief Performs lane-wise inequality comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator!=(const x8float& other) const;

	/**
	* @brief Performs lane-wise less-than comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator<(const x8float& other) const;

	/**
	* @brief Performs lane-wise less-than-or-equal comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator<=(const x8float& other) const;

	/**
	* @brief Performs lane-wise greater-than comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator>(const x8float& other) const;

	/**
	* @brief Performs lane-wise greater-than-or-equal comparison.
	* @param other The values to compare with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator>=(const x8float& other) const;

	/*
	* Bitwise operators
	*/

	/**
	* @brief Performs bitwise AND operation on each lane.
	* @param other The values to AND with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator&(const x8float& other) const;

	/**
	* @brief Performs bitwise OR operation on each lane.
	* @param other The values to OR with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator|(const x8float& other) const;

	/**
	* @brief Performs bitwise XOR operation on each lane.
	* @param other The values to XOR with.
	* @return Each lane contains the boolean value of the operation.
	*/
	x8float operator^(const x8float& other) const;

	/*
	* Compound operators
	*/

	/**
	* @brief Adds a value to this value for each lane.
	* @param other The values to add.
	* @return Reference to self.
	*/
	x8float& operator+=(const x8float& other);

	/**
	* @brief Subtracts a value to this value for each lane.
	* @param other The values to subtract.
	* @return Reference to self.
	*/
	x8float& operator-=(const x8float& other);

	/**
	* @brief Multiplies a value to this value for each lane.
	* @param other The values to multiply.
	* @return Reference to self.
	*/
	x8float& operator*=(const x8float& other);

	/**
	* @brief Divides this value by a value for each lane.
	* @param other The values of the divisors.
	* @return Reference to self.
	*/
	x8float& operator/=(const x8float& other);

	/**
	* @brief Perform bitwise AND with a value for each lane.
	* @param other The values to AND with.
	* @return Reference to self.
	*/
	x8float& operator&=(const x8float& other);

	/**
	* @brief Perform bitwise OR with a value for each lane.
	* @param other The values to OR with.
	* @return Reference to self.
	*/
	x8float& operator|=(const x8float& other);

	/**
	* @brief Perform bitwise XOR with a value for each lane.
	* @param other The values to XOR with.
	* @return Reference to self.
	*/
	x8float& operator^=(const x8float& other);

	/*
	* Stream operators
	*/

	/**
	* @brief Prints the value of this object to a stream.
	* @param stream Stream to print to.
	* @param value Value to print.
	* @return Reference to stream.
	*/
	friend std::ostream& operator<<(std::ostream& stream, const x8float& value);

	/**
	* @brief Reads a value for each lane from stream.
	* @param stream Stream to read from.
	* @param value Value to read to.
	* @return Reference to stream.
	*/
	friend std::istream& operator >> (std::istream& stream, x8float& value);

	/*
	* Element access
	*/

	/**
	* @brief Gets the value of a specified lane.
	* @param i Lane index
	* @return Value of lane.
	*/
	const float& operator[](int i) const;

	/**
	* @brief Gets the value of a specified lane.
	* @param i Lane index
	* @return Value of lane.
	*/
	float& operator[](int i);

	/**
	* @brief Gets the value of a specified lane.
	* @param i Lane index.
	* @return Value of lane.
	*/
	float at(int i) const;

	/*
	* Functions
	*/

	/**
	* @brief Calculates the maximum for each lane.
	* @param a First operand.
	* @param b Second operand.
	* @return Maximum value of each lane.
	*/
	friend x8float max(const x8float& a, const x8float& b);

	/**
	* @brief Calculates the minimum for each lane.
	* @param a First operand.
	* @param b Second operand.
	* @return Minimum value of each lane.
	*/
	friend x8float min(const x8float& a, const x8float& b);

	/**
	* @brief Calculates the reciprocal for each lane.
	* @param a Operand.
	* @return Reciprocal for each lane.
	*/
	friend x8float rcp(const x8float& a);

	/**
	* @brief Calculates the square root for each lane.
	* @param a Operand.
	* @return Square root of each lane.
	*/
	friend x8float sqrt(const x8float& a);

	/**
	* @brief Calculates the reciprocal square root for each lane.
	* @param a Operand.
	* @return Reciprocal square roor for each lane.
	*/
	friend x8float rsqrt(const x8float& a);

	/**
	* @brief Adds odd lane indices and subtracts even lane indices.
	* @param a First operand.
	* @param b Second operand.
	* @return Result of the operation.
	*/
	friend x8float addsub(const x8float& a, const x8float& b);

	/**
	* @brief Rounds the value of each lane up to an integer.
	* @param a Operand.
	* @return Result of the operation.
	*/
	friend x8float ceil(const x8float& a);

	/**
	* @brief Rounds the value of each lane down to an integer.
	* @param a Operand.
	* @return Result of the operation.
	*/
	friend x8float floor(const x8float& a);

	/**
	* @brief Horizontally add adjacent pairs.
	*
	* ret[0] = a[0] + a[1]
	* ret[1] = a[2] + a[3]
	* ret[2] = b[0] + b[1]
	* ret[3] = b[2] + b[3]
	* ret[4] = a[4] + a[5]
	* ret[5] = a[6] + a[7]
	* ret[6] = b[4] + b[5]
	* ret[7] = b[6] + b[7]
	*
	* @param a First operand.
	* @param b Second operand.
	* @return Result of the operation.
	*/
	friend x8float hadd(const x8float& a, const x8float& b);

	/**
	* @brief Horizontally subtract adjacent pairs.
	*
	* ret[0] = a[0] - a[1]
	* ret[1] = a[2] - a[3]
	* ret[2] = b[0] - b[1]
	* ret[3] = b[2] - b[3]
	* ret[4] = a[4] - a[5]
	* ret[5] = a[6] - a[7]
	* ret[6] = b[4] - b[5]
	* ret[7] = b[6] - b[7]
	*
	* @param a First operand.
	* @param b Second operand.
	* @return Result of the operation.
	*/
	friend x8float hsub(const x8float& a, const x8float& b);

	/**
	 * @brief Blends a and b depending on integer mask.
	 *
	 * ret[i] = (mask & (1 << i)) ? a[i] : b[i]
	 *
	 * @param a First operand.
	 * @param b Second operand.
	 * @param mask Control mask.
	 * @return Result of the operation.
	 */
	friend x8float blend(const x8float& a, const x8float& b, const int mask);

	/**
	* @brief Blends a and b depending on lane mask.
	*
	* ret[i] = (mask[i]) ? a[i] : b[i]
	*
	* @param a First operand.
	* @param b Second operand.
	* @param mask Control mask.
	* @return Result of the operation.
	*/
	friend x8float blend(const x8float& a, const x8float& b, const x8float& mask);

	/**
	* @brief Swaps the contents of two objects.
	* @param a First operand.
	* @param b Second operand.
	*/
	friend void swap(x8float& a, x8float& b) noexcept;

	/**
	* @brief The values of each lane.
	*/
	__m256 _data;
};

inline x8float::x8float()
{
	_data = _mm256_setzero_ps();
}

inline x8float::x8float(const float x)
{
	_data = _mm256_set1_ps(x);
}

inline x8float::x8float(
	const float a,
	const float b,
	const float c,
	const float d,
	const float e,
	const float f,
	const float g,
	const float h)
{
	_data = _mm256_setr_ps(a, b, c, d, e, f, g, h);
}

inline x8float::x8float(const float *vec)
{
	_data = _mm256_loadu_ps(vec);
}

inline x8float::x8float(const x4float &a, const x4float &b)
{
	_data = _mm256_setr_m128(a._data, b._data);
}

inline x8float::x8float(const x8float &other)
{
	_data = _mm256_load_ps(other._data.m256_f32);
}

inline x8float::x8float(x8float &&other) noexcept
{
	swap(*this, other);
}

inline x8float::~x8float()
{
	// Do nothing
}

inline x8float & x8float::operator=(const float x)
{
	_data = _mm256_set1_ps(x);

	return *this;
}

inline x8float & x8float::operator=(const float *vec)
{
	_data = _mm256_loadu_ps(vec);

	return *this;
}

inline x8float & x8float::operator=(const x8float &other)
{
	_data = _mm256_load_ps(other._data.m256_f32);

	return *this;
}

inline x8float & x8float::operator=(x8float &&other) noexcept
{
	swap(*this, other);

	return *this;
}

inline x8float x8float::operator+(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_add_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator-(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_sub_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator*(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_mul_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator/(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_div_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator+() const
{
	x8float ret;
	const x8float factor{ 1.0f };

	ret._data = _mm256_mul_ps(_data, factor._data);

	return ret;
}

inline x8float x8float::operator-() const
{
	x8float ret;
	const x8float factor{ -1.0f };

	ret._data = _mm256_mul_ps(_data, factor._data);

	return ret;
}

inline x8float x8float::operator==(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_EQ_OQ);

	return ret;
}

inline x8float x8float::operator!=(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_NEQ_OQ);

	return ret;
}

inline x8float x8float::operator<(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_LT_OQ);

	return ret;
}

inline x8float x8float::operator<=(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_LE_OQ);

	return ret;
}

inline x8float x8float::operator>(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_GT_OQ);

	return ret;
}

inline x8float x8float::operator>=(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_cmp_ps(_data, other._data, _CMP_GE_OQ);

	return ret;
}

inline x8float x8float::operator&(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_and_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator|(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_or_ps(_data, other._data);

	return ret;
}

inline x8float x8float::operator^(const x8float &other) const
{
	x8float ret;

	ret._data = _mm256_xor_ps(_data, other._data);

	return ret;
}

inline x8float & x8float::operator+=(const x8float &other)
{
	_data = _mm256_add_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator-=(const x8float &other)
{
	_data = _mm256_sub_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator*=(const x8float &other)
{
	_data = _mm256_mul_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator/=(const x8float &other)
{
	_data = _mm256_div_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator&=(const x8float &other)
{
	_data = _mm256_and_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator|=(const x8float &other)
{
	_data = _mm256_or_ps(_data, other._data);

	return *this;
}

inline x8float & x8float::operator^=(const x8float &other)
{
	_data = _mm256_xor_ps(_data, other._data);

	return *this;
}

inline const float& x8float::operator[](const int i) const
{
	return _data.m256_f32[i];
}

inline float& x8float::operator[](const int i)
{
	return _data.m256_f32[i];
}

inline float x8float::at(const int i) const
{
	const float ret = _data.m256_f32[i];

	return ret;
}

inline std::ostream & operator<<(std::ostream &stream, const x8float &value)
{
	stream << value[0] << " " << value[1] << " " << value[2] << " " << value[3] << " " << value[4] << " " << value[5] << " " << value[6] << " " << value[7];

	return stream;
}

inline std::istream & operator >> (std::istream &stream, x8float &value)
{
	float f0;
	float f1;
	float f2;
	float f3;
	float f4;
	float f5;
	float f6;
	float f7;

	stream >> f0 >> f1 >> f2 >> f3 >> f4 >> f5 >> f6 >> f7;

	value = x8float{ f0, f1, f2, f3, f4, f5, f6, f7 };

	return stream;
}

inline x8float max(const x8float &a, const x8float &b)
{
	x8float ret;

	ret._data = _mm256_max_ps(a._data, b._data);

	return ret;
}

inline x8float min(const x8float &a, const x8float &b)
{
	x8float ret;

	ret._data = _mm256_min_ps(a._data, b._data);

	return ret;
}

inline x8float rcp(const x8float &a)
{
	x8float ret;

	ret._data = _mm256_rcp_ps(a._data);

	return ret;
}

inline x8float sqrt(const x8float &a)
{
	x8float ret;

	ret._data = _mm256_sqrt_ps(a._data);

	return ret;
}

inline x8float rsqrt(const x8float &a)
{
	x8float ret;

	ret._data = _mm256_rsqrt_ps(a._data);

	return ret;
}

inline x8float addsub(const x8float &a, const x8float &b)
{
	x8float ret;

	ret._data = _mm256_addsub_ps(a._data, b._data);

	return ret;
}

inline x8float ceil(const x8float &a)
{
	x8float ret;

	ret._data = _mm256_ceil_ps(a._data);

	return ret;
}

inline x8float floor(const x8float &a)
{
	x8float ret;

	ret._data = _mm256_floor_ps(a._data);

	return ret;
}

inline x8float hadd(const x8float &a, const x8float &b)
{
	x8float ret;

	ret._data = _mm256_hadd_ps(a._data, b._data);

	return ret;
}

inline x8float hsub(const x8float &a, const x8float &b)
{
	x8float ret;

	ret._data = _mm256_hsub_ps(a._data, b._data);

	return ret;
}

inline x8float blend(const x8float &a, const x8float &b, const int mask)
{
	x8float ret;

	switch (mask & 0xFF)
	{
	case 0:
		ret._data = _mm256_blend_ps(a._data, b._data, 0);
		break;
	case 1:
		ret._data = _mm256_blend_ps(a._data, b._data, 1);
		break;
	case 2:
		ret._data = _mm256_blend_ps(a._data, b._data, 2);
		break;
	case 3:
		ret._data = _mm256_blend_ps(a._data, b._data, 3);
		break;
	case 4:
		ret._data = _mm256_blend_ps(a._data, b._data, 4);
		break;
	case 5:
		ret._data = _mm256_blend_ps(a._data, b._data, 5);
		break;
	case 6:
		ret._data = _mm256_blend_ps(a._data, b._data, 6);
		break;
	case 7:
		ret._data = _mm256_blend_ps(a._data, b._data, 7);
		break;
	case 8:
		ret._data = _mm256_blend_ps(a._data, b._data, 8);
		break;
	case 9:
		ret._data = _mm256_blend_ps(a._data, b._data, 9);
		break;
	case 10:
		ret._data = _mm256_blend_ps(a._data, b._data, 10);
		break;
	case 11:
		ret._data = _mm256_blend_ps(a._data, b._data, 11);
		break;
	case 12:
		ret._data = _mm256_blend_ps(a._data, b._data, 12);
		break;
	case 13:
		ret._data = _mm256_blend_ps(a._data, b._data, 13);
		break;
	case 14:
		ret._data = _mm256_blend_ps(a._data, b._data, 14);
		break;
	case 15:
		ret._data = _mm256_blend_ps(a._data, b._data, 15);
		break;
	case 16:
		ret._data = _mm256_blend_ps(a._data, b._data, 16);
		break;
	case 17:
		ret._data = _mm256_blend_ps(a._data, b._data, 17);
		break;
	case 18:
		ret._data = _mm256_blend_ps(a._data, b._data, 18);
		break;
	case 19:
		ret._data = _mm256_blend_ps(a._data, b._data, 19);
		break;
	case 20:
		ret._data = _mm256_blend_ps(a._data, b._data, 20);
		break;
	case 21:
		ret._data = _mm256_blend_ps(a._data, b._data, 21);
		break;
	case 22:
		ret._data = _mm256_blend_ps(a._data, b._data, 22);
		break;
	case 23:
		ret._data = _mm256_blend_ps(a._data, b._data, 23);
		break;
	case 24:
		ret._data = _mm256_blend_ps(a._data, b._data, 24);
		break;
	case 25:
		ret._data = _mm256_blend_ps(a._data, b._data, 25);
		break;
	case 26:
		ret._data = _mm256_blend_ps(a._data, b._data, 26);
		break;
	case 27:
		ret._data = _mm256_blend_ps(a._data, b._data, 27);
		break;
	case 28:
		ret._data = _mm256_blend_ps(a._data, b._data, 28);
		break;
	case 29:
		ret._data = _mm256_blend_ps(a._data, b._data, 29);
		break;
	case 30:
		ret._data = _mm256_blend_ps(a._data, b._data, 30);
		break;
	case 31:
		ret._data = _mm256_blend_ps(a._data, b._data, 31);
		break;
	case 32:
		ret._data = _mm256_blend_ps(a._data, b._data, 32);
		break;
	case 33:
		ret._data = _mm256_blend_ps(a._data, b._data, 33);
		break;
	case 34:
		ret._data = _mm256_blend_ps(a._data, b._data, 34);
		break;
	case 35:
		ret._data = _mm256_blend_ps(a._data, b._data, 35);
		break;
	case 36:
		ret._data = _mm256_blend_ps(a._data, b._data, 36);
		break;
	case 37:
		ret._data = _mm256_blend_ps(a._data, b._data, 37);
		break;
	case 38:
		ret._data = _mm256_blend_ps(a._data, b._data, 38);
		break;
	case 39:
		ret._data = _mm256_blend_ps(a._data, b._data, 39);
		break;
	case 40:
		ret._data = _mm256_blend_ps(a._data, b._data, 40);
		break;
	case 41:
		ret._data = _mm256_blend_ps(a._data, b._data, 41);
		break;
	case 42:
		ret._data = _mm256_blend_ps(a._data, b._data, 42);
		break;
	case 43:
		ret._data = _mm256_blend_ps(a._data, b._data, 43);
		break;
	case 44:
		ret._data = _mm256_blend_ps(a._data, b._data, 44);
		break;
	case 45:
		ret._data = _mm256_blend_ps(a._data, b._data, 45);
		break;
	case 46:
		ret._data = _mm256_blend_ps(a._data, b._data, 46);
		break;
	case 47:
		ret._data = _mm256_blend_ps(a._data, b._data, 47);
		break;
	case 48:
		ret._data = _mm256_blend_ps(a._data, b._data, 48);
		break;
	case 49:
		ret._data = _mm256_blend_ps(a._data, b._data, 49);
		break;
	case 50:
		ret._data = _mm256_blend_ps(a._data, b._data, 50);
		break;
	case 51:
		ret._data = _mm256_blend_ps(a._data, b._data, 51);
		break;
	case 52:
		ret._data = _mm256_blend_ps(a._data, b._data, 52);
		break;
	case 53:
		ret._data = _mm256_blend_ps(a._data, b._data, 53);
		break;
	case 54:
		ret._data = _mm256_blend_ps(a._data, b._data, 54);
		break;
	case 55:
		ret._data = _mm256_blend_ps(a._data, b._data, 55);
		break;
	case 56:
		ret._data = _mm256_blend_ps(a._data, b._data, 56);
		break;
	case 57:
		ret._data = _mm256_blend_ps(a._data, b._data, 57);
		break;
	case 58:
		ret._data = _mm256_blend_ps(a._data, b._data, 58);
		break;
	case 59:
		ret._data = _mm256_blend_ps(a._data, b._data, 59);
		break;
	case 60:
		ret._data = _mm256_blend_ps(a._data, b._data, 60);
		break;
	case 61:
		ret._data = _mm256_blend_ps(a._data, b._data, 61);
		break;
	case 62:
		ret._data = _mm256_blend_ps(a._data, b._data, 62);
		break;
	case 63:
		ret._data = _mm256_blend_ps(a._data, b._data, 63);
		break;
	case 64:
		ret._data = _mm256_blend_ps(a._data, b._data, 64);
		break;
	case 65:
		ret._data = _mm256_blend_ps(a._data, b._data, 65);
		break;
	case 66:
		ret._data = _mm256_blend_ps(a._data, b._data, 66);
		break;
	case 67:
		ret._data = _mm256_blend_ps(a._data, b._data, 67);
		break;
	case 68:
		ret._data = _mm256_blend_ps(a._data, b._data, 68);
		break;
	case 69:
		ret._data = _mm256_blend_ps(a._data, b._data, 69);
		break;
	case 70:
		ret._data = _mm256_blend_ps(a._data, b._data, 70);
		break;
	case 71:
		ret._data = _mm256_blend_ps(a._data, b._data, 71);
		break;
	case 72:
		ret._data = _mm256_blend_ps(a._data, b._data, 72);
		break;
	case 73:
		ret._data = _mm256_blend_ps(a._data, b._data, 73);
		break;
	case 74:
		ret._data = _mm256_blend_ps(a._data, b._data, 74);
		break;
	case 75:
		ret._data = _mm256_blend_ps(a._data, b._data, 75);
		break;
	case 76:
		ret._data = _mm256_blend_ps(a._data, b._data, 76);
		break;
	case 77:
		ret._data = _mm256_blend_ps(a._data, b._data, 77);
		break;
	case 78:
		ret._data = _mm256_blend_ps(a._data, b._data, 78);
		break;
	case 79:
		ret._data = _mm256_blend_ps(a._data, b._data, 79);
		break;
	case 80:
		ret._data = _mm256_blend_ps(a._data, b._data, 80);
		break;
	case 81:
		ret._data = _mm256_blend_ps(a._data, b._data, 81);
		break;
	case 82:
		ret._data = _mm256_blend_ps(a._data, b._data, 82);
		break;
	case 83:
		ret._data = _mm256_blend_ps(a._data, b._data, 83);
		break;
	case 84:
		ret._data = _mm256_blend_ps(a._data, b._data, 84);
		break;
	case 85:
		ret._data = _mm256_blend_ps(a._data, b._data, 85);
		break;
	case 86:
		ret._data = _mm256_blend_ps(a._data, b._data, 86);
		break;
	case 87:
		ret._data = _mm256_blend_ps(a._data, b._data, 87);
		break;
	case 88:
		ret._data = _mm256_blend_ps(a._data, b._data, 88);
		break;
	case 89:
		ret._data = _mm256_blend_ps(a._data, b._data, 89);
		break;
	case 90:
		ret._data = _mm256_blend_ps(a._data, b._data, 90);
		break;
	case 91:
		ret._data = _mm256_blend_ps(a._data, b._data, 91);
		break;
	case 92:
		ret._data = _mm256_blend_ps(a._data, b._data, 92);
		break;
	case 93:
		ret._data = _mm256_blend_ps(a._data, b._data, 93);
		break;
	case 94:
		ret._data = _mm256_blend_ps(a._data, b._data, 94);
		break;
	case 95:
		ret._data = _mm256_blend_ps(a._data, b._data, 95);
		break;
	case 96:
		ret._data = _mm256_blend_ps(a._data, b._data, 96);
		break;
	case 97:
		ret._data = _mm256_blend_ps(a._data, b._data, 97);
		break;
	case 98:
		ret._data = _mm256_blend_ps(a._data, b._data, 98);
		break;
	case 99:
		ret._data = _mm256_blend_ps(a._data, b._data, 99);
		break;
	case 100:
		ret._data = _mm256_blend_ps(a._data, b._data, 100);
		break;
	case 101:
		ret._data = _mm256_blend_ps(a._data, b._data, 101);
		break;
	case 102:
		ret._data = _mm256_blend_ps(a._data, b._data, 102);
		break;
	case 103:
		ret._data = _mm256_blend_ps(a._data, b._data, 103);
		break;
	case 104:
		ret._data = _mm256_blend_ps(a._data, b._data, 104);
		break;
	case 105:
		ret._data = _mm256_blend_ps(a._data, b._data, 105);
		break;
	case 106:
		ret._data = _mm256_blend_ps(a._data, b._data, 106);
		break;
	case 107:
		ret._data = _mm256_blend_ps(a._data, b._data, 107);
		break;
	case 108:
		ret._data = _mm256_blend_ps(a._data, b._data, 108);
		break;
	case 109:
		ret._data = _mm256_blend_ps(a._data, b._data, 109);
		break;
	case 110:
		ret._data = _mm256_blend_ps(a._data, b._data, 110);
		break;
	case 111:
		ret._data = _mm256_blend_ps(a._data, b._data, 111);
		break;
	case 112:
		ret._data = _mm256_blend_ps(a._data, b._data, 112);
		break;
	case 113:
		ret._data = _mm256_blend_ps(a._data, b._data, 113);
		break;
	case 114:
		ret._data = _mm256_blend_ps(a._data, b._data, 114);
		break;
	case 115:
		ret._data = _mm256_blend_ps(a._data, b._data, 115);
		break;
	case 116:
		ret._data = _mm256_blend_ps(a._data, b._data, 116);
		break;
	case 117:
		ret._data = _mm256_blend_ps(a._data, b._data, 117);
		break;
	case 118:
		ret._data = _mm256_blend_ps(a._data, b._data, 118);
		break;
	case 119:
		ret._data = _mm256_blend_ps(a._data, b._data, 119);
		break;
	case 120:
		ret._data = _mm256_blend_ps(a._data, b._data, 120);
		break;
	case 121:
		ret._data = _mm256_blend_ps(a._data, b._data, 121);
		break;
	case 122:
		ret._data = _mm256_blend_ps(a._data, b._data, 122);
		break;
	case 123:
		ret._data = _mm256_blend_ps(a._data, b._data, 123);
		break;
	case 124:
		ret._data = _mm256_blend_ps(a._data, b._data, 124);
		break;
	case 125:
		ret._data = _mm256_blend_ps(a._data, b._data, 125);
		break;
	case 126:
		ret._data = _mm256_blend_ps(a._data, b._data, 126);
		break;
	case 127:
		ret._data = _mm256_blend_ps(a._data, b._data, 127);
		break;
	case 128:
		ret._data = _mm256_blend_ps(a._data, b._data, 128);
		break;
	case 129:
		ret._data = _mm256_blend_ps(a._data, b._data, 129);
		break;
	case 130:
		ret._data = _mm256_blend_ps(a._data, b._data, 130);
		break;
	case 131:
		ret._data = _mm256_blend_ps(a._data, b._data, 131);
		break;
	case 132:
		ret._data = _mm256_blend_ps(a._data, b._data, 132);
		break;
	case 133:
		ret._data = _mm256_blend_ps(a._data, b._data, 133);
		break;
	case 134:
		ret._data = _mm256_blend_ps(a._data, b._data, 134);
		break;
	case 135:
		ret._data = _mm256_blend_ps(a._data, b._data, 135);
		break;
	case 136:
		ret._data = _mm256_blend_ps(a._data, b._data, 136);
		break;
	case 137:
		ret._data = _mm256_blend_ps(a._data, b._data, 137);
		break;
	case 138:
		ret._data = _mm256_blend_ps(a._data, b._data, 138);
		break;
	case 139:
		ret._data = _mm256_blend_ps(a._data, b._data, 139);
		break;
	case 140:
		ret._data = _mm256_blend_ps(a._data, b._data, 140);
		break;
	case 141:
		ret._data = _mm256_blend_ps(a._data, b._data, 141);
		break;
	case 142:
		ret._data = _mm256_blend_ps(a._data, b._data, 142);
		break;
	case 143:
		ret._data = _mm256_blend_ps(a._data, b._data, 143);
		break;
	case 144:
		ret._data = _mm256_blend_ps(a._data, b._data, 144);
		break;
	case 145:
		ret._data = _mm256_blend_ps(a._data, b._data, 145);
		break;
	case 146:
		ret._data = _mm256_blend_ps(a._data, b._data, 146);
		break;
	case 147:
		ret._data = _mm256_blend_ps(a._data, b._data, 147);
		break;
	case 148:
		ret._data = _mm256_blend_ps(a._data, b._data, 148);
		break;
	case 149:
		ret._data = _mm256_blend_ps(a._data, b._data, 149);
		break;
	case 150:
		ret._data = _mm256_blend_ps(a._data, b._data, 150);
		break;
	case 151:
		ret._data = _mm256_blend_ps(a._data, b._data, 151);
		break;
	case 152:
		ret._data = _mm256_blend_ps(a._data, b._data, 152);
		break;
	case 153:
		ret._data = _mm256_blend_ps(a._data, b._data, 153);
		break;
	case 154:
		ret._data = _mm256_blend_ps(a._data, b._data, 154);
		break;
	case 155:
		ret._data = _mm256_blend_ps(a._data, b._data, 155);
		break;
	case 156:
		ret._data = _mm256_blend_ps(a._data, b._data, 156);
		break;
	case 157:
		ret._data = _mm256_blend_ps(a._data, b._data, 157);
		break;
	case 158:
		ret._data = _mm256_blend_ps(a._data, b._data, 158);
		break;
	case 159:
		ret._data = _mm256_blend_ps(a._data, b._data, 159);
		break;
	case 160:
		ret._data = _mm256_blend_ps(a._data, b._data, 160);
		break;
	case 161:
		ret._data = _mm256_blend_ps(a._data, b._data, 161);
		break;
	case 162:
		ret._data = _mm256_blend_ps(a._data, b._data, 162);
		break;
	case 163:
		ret._data = _mm256_blend_ps(a._data, b._data, 163);
		break;
	case 164:
		ret._data = _mm256_blend_ps(a._data, b._data, 164);
		break;
	case 165:
		ret._data = _mm256_blend_ps(a._data, b._data, 165);
		break;
	case 166:
		ret._data = _mm256_blend_ps(a._data, b._data, 166);
		break;
	case 167:
		ret._data = _mm256_blend_ps(a._data, b._data, 167);
		break;
	case 168:
		ret._data = _mm256_blend_ps(a._data, b._data, 168);
		break;
	case 169:
		ret._data = _mm256_blend_ps(a._data, b._data, 169);
		break;
	case 170:
		ret._data = _mm256_blend_ps(a._data, b._data, 170);
		break;
	case 171:
		ret._data = _mm256_blend_ps(a._data, b._data, 171);
		break;
	case 172:
		ret._data = _mm256_blend_ps(a._data, b._data, 172);
		break;
	case 173:
		ret._data = _mm256_blend_ps(a._data, b._data, 173);
		break;
	case 174:
		ret._data = _mm256_blend_ps(a._data, b._data, 174);
		break;
	case 175:
		ret._data = _mm256_blend_ps(a._data, b._data, 175);
		break;
	case 176:
		ret._data = _mm256_blend_ps(a._data, b._data, 176);
		break;
	case 177:
		ret._data = _mm256_blend_ps(a._data, b._data, 177);
		break;
	case 178:
		ret._data = _mm256_blend_ps(a._data, b._data, 178);
		break;
	case 179:
		ret._data = _mm256_blend_ps(a._data, b._data, 179);
		break;
	case 180:
		ret._data = _mm256_blend_ps(a._data, b._data, 180);
		break;
	case 181:
		ret._data = _mm256_blend_ps(a._data, b._data, 181);
		break;
	case 182:
		ret._data = _mm256_blend_ps(a._data, b._data, 182);
		break;
	case 183:
		ret._data = _mm256_blend_ps(a._data, b._data, 183);
		break;
	case 184:
		ret._data = _mm256_blend_ps(a._data, b._data, 184);
		break;
	case 185:
		ret._data = _mm256_blend_ps(a._data, b._data, 185);
		break;
	case 186:
		ret._data = _mm256_blend_ps(a._data, b._data, 186);
		break;
	case 187:
		ret._data = _mm256_blend_ps(a._data, b._data, 187);
		break;
	case 188:
		ret._data = _mm256_blend_ps(a._data, b._data, 188);
		break;
	case 189:
		ret._data = _mm256_blend_ps(a._data, b._data, 189);
		break;
	case 190:
		ret._data = _mm256_blend_ps(a._data, b._data, 190);
		break;
	case 191:
		ret._data = _mm256_blend_ps(a._data, b._data, 191);
		break;
	case 192:
		ret._data = _mm256_blend_ps(a._data, b._data, 192);
		break;
	case 193:
		ret._data = _mm256_blend_ps(a._data, b._data, 193);
		break;
	case 194:
		ret._data = _mm256_blend_ps(a._data, b._data, 194);
		break;
	case 195:
		ret._data = _mm256_blend_ps(a._data, b._data, 195);
		break;
	case 196:
		ret._data = _mm256_blend_ps(a._data, b._data, 196);
		break;
	case 197:
		ret._data = _mm256_blend_ps(a._data, b._data, 197);
		break;
	case 198:
		ret._data = _mm256_blend_ps(a._data, b._data, 198);
		break;
	case 199:
		ret._data = _mm256_blend_ps(a._data, b._data, 199);
		break;
	case 200:
		ret._data = _mm256_blend_ps(a._data, b._data, 200);
		break;
	case 201:
		ret._data = _mm256_blend_ps(a._data, b._data, 201);
		break;
	case 202:
		ret._data = _mm256_blend_ps(a._data, b._data, 202);
		break;
	case 203:
		ret._data = _mm256_blend_ps(a._data, b._data, 203);
		break;
	case 204:
		ret._data = _mm256_blend_ps(a._data, b._data, 204);
		break;
	case 205:
		ret._data = _mm256_blend_ps(a._data, b._data, 205);
		break;
	case 206:
		ret._data = _mm256_blend_ps(a._data, b._data, 206);
		break;
	case 207:
		ret._data = _mm256_blend_ps(a._data, b._data, 207);
		break;
	case 208:
		ret._data = _mm256_blend_ps(a._data, b._data, 208);
		break;
	case 209:
		ret._data = _mm256_blend_ps(a._data, b._data, 209);
		break;
	case 210:
		ret._data = _mm256_blend_ps(a._data, b._data, 210);
		break;
	case 211:
		ret._data = _mm256_blend_ps(a._data, b._data, 211);
		break;
	case 212:
		ret._data = _mm256_blend_ps(a._data, b._data, 212);
		break;
	case 213:
		ret._data = _mm256_blend_ps(a._data, b._data, 213);
		break;
	case 214:
		ret._data = _mm256_blend_ps(a._data, b._data, 214);
		break;
	case 215:
		ret._data = _mm256_blend_ps(a._data, b._data, 215);
		break;
	case 216:
		ret._data = _mm256_blend_ps(a._data, b._data, 216);
		break;
	case 217:
		ret._data = _mm256_blend_ps(a._data, b._data, 217);
		break;
	case 218:
		ret._data = _mm256_blend_ps(a._data, b._data, 218);
		break;
	case 219:
		ret._data = _mm256_blend_ps(a._data, b._data, 219);
		break;
	case 220:
		ret._data = _mm256_blend_ps(a._data, b._data, 220);
		break;
	case 221:
		ret._data = _mm256_blend_ps(a._data, b._data, 221);
		break;
	case 222:
		ret._data = _mm256_blend_ps(a._data, b._data, 222);
		break;
	case 223:
		ret._data = _mm256_blend_ps(a._data, b._data, 223);
		break;
	case 224:
		ret._data = _mm256_blend_ps(a._data, b._data, 224);
		break;
	case 225:
		ret._data = _mm256_blend_ps(a._data, b._data, 225);
		break;
	case 226:
		ret._data = _mm256_blend_ps(a._data, b._data, 226);
		break;
	case 227:
		ret._data = _mm256_blend_ps(a._data, b._data, 227);
		break;
	case 228:
		ret._data = _mm256_blend_ps(a._data, b._data, 228);
		break;
	case 229:
		ret._data = _mm256_blend_ps(a._data, b._data, 229);
		break;
	case 230:
		ret._data = _mm256_blend_ps(a._data, b._data, 230);
		break;
	case 231:
		ret._data = _mm256_blend_ps(a._data, b._data, 231);
		break;
	case 232:
		ret._data = _mm256_blend_ps(a._data, b._data, 232);
		break;
	case 233:
		ret._data = _mm256_blend_ps(a._data, b._data, 233);
		break;
	case 234:
		ret._data = _mm256_blend_ps(a._data, b._data, 234);
		break;
	case 235:
		ret._data = _mm256_blend_ps(a._data, b._data, 235);
		break;
	case 236:
		ret._data = _mm256_blend_ps(a._data, b._data, 236);
		break;
	case 237:
		ret._data = _mm256_blend_ps(a._data, b._data, 237);
		break;
	case 238:
		ret._data = _mm256_blend_ps(a._data, b._data, 238);
		break;
	case 239:
		ret._data = _mm256_blend_ps(a._data, b._data, 239);
		break;
	case 240:
		ret._data = _mm256_blend_ps(a._data, b._data, 240);
		break;
	case 241:
		ret._data = _mm256_blend_ps(a._data, b._data, 241);
		break;
	case 242:
		ret._data = _mm256_blend_ps(a._data, b._data, 242);
		break;
	case 243:
		ret._data = _mm256_blend_ps(a._data, b._data, 243);
		break;
	case 244:
		ret._data = _mm256_blend_ps(a._data, b._data, 244);
		break;
	case 245:
		ret._data = _mm256_blend_ps(a._data, b._data, 245);
		break;
	case 246:
		ret._data = _mm256_blend_ps(a._data, b._data, 246);
		break;
	case 247:
		ret._data = _mm256_blend_ps(a._data, b._data, 247);
		break;
	case 248:
		ret._data = _mm256_blend_ps(a._data, b._data, 248);
		break;
	case 249:
		ret._data = _mm256_blend_ps(a._data, b._data, 249);
		break;
	case 250:
		ret._data = _mm256_blend_ps(a._data, b._data, 250);
		break;
	case 251:
		ret._data = _mm256_blend_ps(a._data, b._data, 251);
		break;
	case 252:
		ret._data = _mm256_blend_ps(a._data, b._data, 252);
		break;
	case 253:
		ret._data = _mm256_blend_ps(a._data, b._data, 253);
		break;
	case 254:
		ret._data = _mm256_blend_ps(a._data, b._data, 254);
		break;
	case 255:
		ret._data = _mm256_blend_ps(a._data, b._data, 255);
		break;
	default:
		break;
	}

	return ret;
}

inline x8float blend(const x8float &a, const x8float &b, const x8float &mask)
{
	x8float ret;


	ret._data = _mm256_blendv_ps(a._data, b._data, mask._data);

	return ret;
}

inline void swap(x8float &a, x8float &b) noexcept
{
	const __m256 temp = _mm256_load_ps(a._data.m256_f32);

	a._data = _mm256_load_ps(b._data.m256_f32);
	b._data = _mm256_load_ps(temp.m256_f32);
}
