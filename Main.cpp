#include <iostream>

#include <xmmintrin.h>

#include <math.h>
#include <time.h>

#include "SIMD.h"

#include <gtest/gtest.h>

#include <intrin.h>

#include <random>
#include <limits>

void checkForCapabilities()
{

	int cpuinfo[4];
	__cpuid(cpuinfo, 1);

	// Check SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 support
	bool sseSupportted = cpuinfo[3] & 1 << 25;
	bool sse2Supportted = cpuinfo[3] & 1 << 26;
	bool sse3Supportted = cpuinfo[2] & 1 << 0;
	bool ssse3Supportted = cpuinfo[2] & 1 << 9;
	bool sse4_1Supportted = cpuinfo[2] & 1 << 19;
	bool sse4_2Supportted = cpuinfo[2] & 1 << 20;

	bool avxSupportted = cpuinfo[2] & 1 << 28;
	bool osxsaveSupported = cpuinfo[2] & 1 << 27;
	if (osxsaveSupported && avxSupportted)
	{
		// _XCR_XFEATURE_ENABLED_MASK = 0
		unsigned long long xcrFeatureMask = _xgetbv(0);
		avxSupportted = (xcrFeatureMask & 0x6) == 0x6;
	}
	// ----------------------------------------------------------------------

	std::cout << "SSE:" << (sseSupportted ? 1 : 0) << std::endl;
	std::cout << "SSE2:" << (sse2Supportted ? 1 : 0) << std::endl;
	std::cout << "SSE3:" << (sse3Supportted ? 1 : 0) << std::endl;
	std::cout << "SSSE3:" << (ssse3Supportted ? 1 : 0) << std::endl;
	std::cout << "SSE4.1:" << (sse4_1Supportted ? 1 : 0) << std::endl;
	std::cout << "SSE4.2:" << (sse4_2Supportted ? 1 : 0) << std::endl;
	std::cout << "AVX:" << (avxSupportted ? 1 : 0) << std::endl;
}

void runComparison()
{
	printf("Starting SIMD 128 bit calculation...\n");

	const int length = 64000;


	float *pResult = (float *)_aligned_malloc(length * sizeof(float), 32);

	__m128 x128;
	__m128 xDelta128 = _mm_set1_ps(4.0f);
	__m128 *pResultSSE128 = (__m128 *) pResult;

	const int SSELength128 = length / 4;

	clock_t t;

	t = clock();

	for (int stress = 0; stress < 1000000; ++stress)
	{
		x128 = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);

		for (int i = 0; i < SSELength128; ++i)
		{
			__m128 xSqrt128 = _mm_sqrt_ps(x128);
			__m128 xRecip128 = _mm_rcp_ps(x128);

			pResultSSE128[i] = _mm_mul_ps(xRecip128, xSqrt128);

			x128 = _mm_add_ps(x128, xDelta128);
		}
	}

	t = clock() - t;

	double time_taken = (double)t / CLOCKS_PER_SEC;

	for (int i = 0; i < 20; ++i)
	{
		//printf("Result[%d] = %f\n", i, pResult[i]);
	}

	printf("Completed SIMD 128 bit Calculation. Time spent: %.2fs\n", time_taken);

	printf("Starting SIMD 256 bit calculation...\n");

	__m256 x256;
	__m256 xDelta256 = _mm256_set1_ps(8.0f);
	__m256 *pResultSSE256 = (__m256 *) pResult;

	const int SSELength256 = length / 8;

	t = clock();

	for (int stress = 0; stress < 1000000; ++stress)
	{
		x256 = _mm256_set_ps(8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);

		for (int i = 0; i < SSELength256; ++i)
		{
			__m256 xSqrt256 = _mm256_sqrt_ps(x256);
			__m256 xRecip256 = _mm256_rcp_ps(x256);

			pResultSSE256[i] = _mm256_mul_ps(xSqrt256, xRecip256);

			x256 = _mm256_add_ps(x256, xDelta256);
		}
	}

	t = clock() - t;

	time_taken = (double)t / CLOCKS_PER_SEC;

	printf("Completed SIMD 256 bit Calculation. Time spent: %.2fs\n", time_taken);

	printf("Starting scalar calculation...\n");

	t = clock();

	for (int stress = 0; stress < 1000000; ++stress)
	{
		float xFloat = 1.0f;

		for (int i = 0; i < length; ++i)
		{
			pResult[i] = sqrt(xFloat) / xFloat;
			xFloat += 1.0f;
		}
	}

	t = clock() - t;

	time_taken = (double)t / CLOCKS_PER_SEC;

	printf("Completed Scalar Calculation. Time spent: %.2fs\n", time_taken);
}

void runx4floatFunctionTest()
{
	std::cout << "Running x4float Function Test." << std::endl;

	x4float a{ 1.5f, 2.23f, 3.37f, 4.82f };
	x4float b{ 1.0f, 3.0f, 2.0f, 0.0f };

	// TODO: Write better test functions/framework to make sure precision of the
	// functions are ok.
	x4float c = max(a, b);
	x4float d = min(a, b);
	x4float e = rcp(a);
	x4float f = sqrt(a);
	x4float g = rsqrt(a);
	x4float h = addsub(a, b);
	x4float i = ceil(a);
	x4float j = floor(a);
	x4float k = hadd(a, b);
	x4float l = hsub(a, b);
	x4float m = mod(a, b);
	x4float sin_val = sin(a);
	x4float cos_val = cos(a);
	x4float tan_val = tan(a);
	x4float atan_val = atan(a);
	x4float asin_val = asin(a);
	x4float acos_val = acos(a);
	x4float log2_val = log2(a);
	x4float pow2_val = pow2(a);
	x4float exp_val = exp(a);
	x4float pow_val = pow(a, b);
	x4float sinh_val = sinh(a);
	x4float cosh_val = cosh(a);
	x4float tanh_val = tanh(a);
	x4float atanh_val = atanh(a);
	x4float asinh_val = asinh(a);
	x4float acosh_val = acosh(a);
	x4float csc_val = csc(a);
	x4float sec_val = sec(a);
	x4float cot_val = cot(a);
	x4float csch_val = csch(a);
	x4float sech_val = sech(a);
	x4float coth_val = coth(a);
	x4float acsc_val = acsc(a);
	x4float asec_val = asec(a);
	x4float acot_val = acot(a);
	x4float acsch_val = acsch(a);
	x4float asech_val = asech(a);
	x4float acoth_val = acoth(a);
	x4float lgamma_val = lgamma(a);
	x4float digamma_val = digamma(a);

	std::cout << "c: " << c << std::endl;
	std::cout << "d: " << d << std::endl;
	std::cout << "e: " << e << std::endl;
	std::cout << "f: " << f << std::endl;
	std::cout << "g: " << g << std::endl;
	std::cout << "h: " << h << std::endl;
	std::cout << "i: " << i << std::endl;
	std::cout << "j: " << j << std::endl;
	std::cout << "k: " << k << std::endl;
	std::cout << "l: " << l << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "sin: " << sin_val << std::endl;
	std::cout << "cos: " << cos_val << std::endl;
	std::cout << "tan: " << tan_val << std::endl;
	std::cout << "atan: " << atan_val << std::endl;
	std::cout << "asin: " << asin_val << std::endl;
	std::cout << "acos: " << acos_val << std::endl;
	std::cout << "log2: " << log2_val << std::endl;
	std::cout << "pow2: " << pow2_val << std::endl;
	std::cout << "exp: " << exp_val << std::endl;
	std::cout << "pow: " << pow_val << std::endl;
	std::cout << "sinh: " << sinh_val << std::endl;
	std::cout << "cosh: " << cosh_val << std::endl;
	std::cout << "tanh: " << tanh_val << std::endl;
	std::cout << "atanh: " << atanh_val << std::endl;
	std::cout << "asinh: " << asinh_val << std::endl;
	std::cout << "acosh: " << acosh_val << std::endl;
	std::cout << "csc: " << csc_val << std::endl;
	std::cout << "sec: " << sec_val << std::endl;
	std::cout << "cot: " << cot_val << std::endl;
	std::cout << "csch: " << csch_val << std::endl;
	std::cout << "sech: " << sech_val << std::endl;
	std::cout << "coth: " << coth_val << std::endl;
	std::cout << "acsc: " << acsc_val << std::endl;
	std::cout << "asec: " << asec_val << std::endl;
	std::cout << "acot: " << acot_val << std::endl;
	std::cout << "acsch: " << acsch_val << std::endl;
	std::cout << "asech: " << asech_val << std::endl;
	std::cout << "acoth: " << acoth_val << std::endl;
	std::cout << "lgamma: " << lgamma_val << std::endl;
	std::cout << "digamma: " << digamma_val << std::endl;

	std::cout << "x4float Function Test complete." << std::endl;
}

void runx4floatMemberAccessTest()
{
	std::cout << "Running x4float Member Access Test." << std::endl;

	x4float a{ 1.0f, 2.0f, 3.0f, 4.0f };

	float f = a[0];

	a[1] = 51.0f;

	std::cout << "f: " << f << std::endl;
	std::cout << "a: " << a << std::endl;

	std::cout << "x4float Member Access Test complete." << std::endl;
}

void runx4floatTests()
{
	runx4floatFunctionTest();
	runx4floatMemberAccessTest();
}

/*
 * New test code
 */

struct floatset
{
	float a1;
	float a2;
	float a3;
	float a4;
	float b1;
	float b2;
	float b3;
	float b4;
};

// TODO: Find a better way to generate float values, preferrably logarithmic scale from MIN to MAX
std::default_random_engine GENERATOR;
std::uniform_real_distribution<float> DIST{ -10.f, 10.f }; 

floatset randomFloatSet()
{
	return floatset{
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR),
		(float)DIST(GENERATOR)
	};
}

std::vector<floatset> genFloatSetArray(int N)
{
	std::vector<floatset> vec;

	vec.reserve(N);

	for (int i = 0; i < N; ++i)
	{
		vec.push_back(randomFloatSet());
	}

	return vec;
}

class x4floatTestFixture : public ::testing::TestWithParam<floatset>
{

};

TEST_P(x4floatTestFixture, Constructor)
{
	float vec[4] = { GetParam().a1, GetParam().a2, GetParam().a3, GetParam().a4 };
	ALIGNED16 float vec2[4] = { GetParam().b1, GetParam().b2, GetParam().b3, GetParam().b4 };

	// Default constructor
	x4float x1;

	// Single float constructor.
	x4float x2{ 1.f };

	// Explicit float constructor.
	x4float x3{ 1.0f, 2.0f, 3.0f, 4.0f };

	// Unaligned vector constructor.
	x4float x4{ vec };

	// Copy constructor
	x4float x5{ x3 };

	// Move constructor
	x4float x6{ +x3 };

	// Aligned vector constructor.
	x4float x7{ vec2 };

	EXPECT_FLOAT_EQ(x1._data.m128_f32[0], 0.f) << "Default constructor should initialize to zero.";
	EXPECT_FLOAT_EQ(x1._data.m128_f32[1], 0.f) << "Default constructor should initialize to zero.";
	EXPECT_FLOAT_EQ(x1._data.m128_f32[2], 0.f) << "Default constructor should initialize to zero.";
	EXPECT_FLOAT_EQ(x1._data.m128_f32[3], 0.f) << "Default constructor should initialize to zero.";

	EXPECT_FLOAT_EQ(x2._data.m128_f32[0], 1.f) << "Single float constructor should initialize all fields to the value of the float.";
	EXPECT_FLOAT_EQ(x2._data.m128_f32[1], 1.f) << "Single float constructor should initialize all fields to the value of the float.";
	EXPECT_FLOAT_EQ(x2._data.m128_f32[2], 1.f) << "Single float constructor should initialize all fields to the value of the float.";
	EXPECT_FLOAT_EQ(x2._data.m128_f32[3], 1.f) << "Single float constructor should initialize all fields to the value of the float.";

	EXPECT_FLOAT_EQ(x3._data.m128_f32[0], 1.f) << "Explicit float constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x3._data.m128_f32[1], 2.f) << "Explicit float constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x3._data.m128_f32[2], 3.f) << "Explicit float constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x3._data.m128_f32[3], 4.f) << "Explicit float constructor should initialize lane to given float value.";

	EXPECT_FLOAT_EQ(x4._data.m128_f32[0], GetParam().a1) << "Unaligned vector constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x4._data.m128_f32[1], GetParam().a2) << "Unaligned vector constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x4._data.m128_f32[2], GetParam().a3) << "Unaligned vector constructor should initialize lane to given float value.";
	EXPECT_FLOAT_EQ(x4._data.m128_f32[3], GetParam().a4) << "Unaligned vector constructor should initialize lane to given float value.";

	EXPECT_FLOAT_EQ(x5._data.m128_f32[0], x3._data.m128_f32[0]);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[1], x3._data.m128_f32[1]);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[2], x3._data.m128_f32[2]);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[3], x3._data.m128_f32[3]);

	EXPECT_FLOAT_EQ(x6._data.m128_f32[0], x3._data.m128_f32[0]);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[1], x3._data.m128_f32[1]);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[2], x3._data.m128_f32[2]);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[3], x3._data.m128_f32[3]);

	EXPECT_FLOAT_EQ(x7._data.m128_f32[0], GetParam().b1);
	EXPECT_FLOAT_EQ(x7._data.m128_f32[1], GetParam().b2);
	EXPECT_FLOAT_EQ(x7._data.m128_f32[2], GetParam().b3);
	EXPECT_FLOAT_EQ(x7._data.m128_f32[3], GetParam().b4);
}

TEST_P(x4floatTestFixture, assignment)
{
	ALIGNED16 float vec2[4] = { GetParam().a1, GetParam().a2, GetParam().a3, GetParam().a4 };

	x4float x1;
	x4float x2;
	x4float x3;
	x4float x4;

	x1 = GetParam().b1;

	x2 = vec2;

	x3 = x1;

	x4 = +x1;

	EXPECT_FLOAT_EQ(x1._data.m128_f32[0], GetParam().b1);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[1], GetParam().b1);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[2], GetParam().b1);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[3], GetParam().b1);

	EXPECT_FLOAT_EQ(x2._data.m128_f32[0], GetParam().a1);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[1], GetParam().a2);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[2], GetParam().a3);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[3], GetParam().a4);

	EXPECT_FLOAT_EQ(x3._data.m128_f32[0], GetParam().b1);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[1], GetParam().b1);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[2], GetParam().b1);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[3], GetParam().b1);

	EXPECT_FLOAT_EQ(x4._data.m128_f32[0], GetParam().b1);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[1], GetParam().b1);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[2], GetParam().b1);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[3], GetParam().b1);
}

TEST_P(x4floatTestFixture, arithmetic)
{
	float a1 = GetParam().a1;
	float a2 = GetParam().a2;
	float a3 = GetParam().a3;
	float a4 = GetParam().a4;
	float b1 = GetParam().b1;
	float b2 = GetParam().b2;
	float b3 = GetParam().b3;
	float b4 = GetParam().b4;

	x4float a{ a1, a2, a3, a4 };
	x4float b{ b1, b2, b3, b4 };

	x4float x1 = a + b;
	x4float x2 = a - b;
	x4float x3 = a * b;
	x4float x4 = a / b;

	x4float x5 = +a;
	x4float x6 = -a;

	EXPECT_FLOAT_EQ(x1._data.m128_f32[0], a1 + b1);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[1], a2 + b2);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[2], a3 + b3);
	EXPECT_FLOAT_EQ(x1._data.m128_f32[3], a4 + b4);

	EXPECT_FLOAT_EQ(x2._data.m128_f32[0], a1 - b1);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[1], a2 - b2);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[2], a3 - b3);
	EXPECT_FLOAT_EQ(x2._data.m128_f32[3], a4 - b4);

	EXPECT_FLOAT_EQ(x3._data.m128_f32[0], a1 * b1);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[1], a2 * b2);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[2], a3 * b3);
	EXPECT_FLOAT_EQ(x3._data.m128_f32[3], a4 * b4);

	EXPECT_FLOAT_EQ(x4._data.m128_f32[0], a1 / b1);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[1], a2 / b2);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[2], a3 / b3);
	EXPECT_FLOAT_EQ(x4._data.m128_f32[3], a4 / b4);

	EXPECT_FLOAT_EQ(x5._data.m128_f32[0], a1);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[1], a2);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[2], a3);
	EXPECT_FLOAT_EQ(x5._data.m128_f32[3], a4);

	EXPECT_FLOAT_EQ(x6._data.m128_f32[0], -a1);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[1], -a2);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[2], -a3);
	EXPECT_FLOAT_EQ(x6._data.m128_f32[3], -a4);
}

TEST(x4float, x4floatComparisonTest)
{
	x4float a{ 1.0f, 2.0f, 3.0f, 4.0f };
	x4float b{ 1.0f, 3.0f, 2.0f, 0.0f };

	x4float x1 = a == b;
	x4float x2 = a != b;
	x4float x3 = a < b;
	x4float x4 = a <= b;
	x4float x5 = a > b;
	x4float x6 = a >= b;

	EXPECT_EQ(x1._data.m128_i32[0], 0xFFFFFFFF);
	EXPECT_EQ(x1._data.m128_i32[1], 0);
	EXPECT_EQ(x1._data.m128_i32[2], 0);
	EXPECT_EQ(x1._data.m128_i32[3], 0);

	EXPECT_EQ(x2._data.m128_i32[0], 0);
	EXPECT_EQ(x2._data.m128_i32[1], 0xFFFFFFFF);
	EXPECT_EQ(x2._data.m128_i32[2], 0xFFFFFFFF);
	EXPECT_EQ(x2._data.m128_i32[3], 0xFFFFFFFF);

	EXPECT_EQ(x3._data.m128_i32[0], 0);
	EXPECT_EQ(x3._data.m128_i32[1], 0xFFFFFFFF);
	EXPECT_EQ(x3._data.m128_i32[2], 0);
	EXPECT_EQ(x3._data.m128_i32[3], 0);

	EXPECT_EQ(x4._data.m128_i32[0], 0xFFFFFFFF);
	EXPECT_EQ(x4._data.m128_i32[1], 0xFFFFFFFF);
	EXPECT_EQ(x4._data.m128_i32[2], 0);
	EXPECT_EQ(x4._data.m128_i32[3], 0);

	EXPECT_EQ(x5._data.m128_i32[0], 0);
	EXPECT_EQ(x5._data.m128_i32[1], 0);
	EXPECT_EQ(x5._data.m128_i32[2], 0xFFFFFFFF);
	EXPECT_EQ(x5._data.m128_i32[3], 0xFFFFFFFF);

	EXPECT_EQ(x6._data.m128_i32[0], 0xFFFFFFFF);
	EXPECT_EQ(x6._data.m128_i32[1], 0);
	EXPECT_EQ(x6._data.m128_i32[2], 0xFFFFFFFF);
	EXPECT_EQ(x6._data.m128_i32[3], 0xFFFFFFFF);
}

INSTANTIATE_TEST_CASE_P(x4float,
	x4floatTestFixture,
	::testing::ValuesIn(genFloatSetArray(10)));

int main(int argc, char **argv)
{
	testing::InitGoogleTest(&argc, argv);

	srand(time(NULL));

	return RUN_ALL_TESTS();
}