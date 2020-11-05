#ifndef CVTX_UINTKEY64_H
#define CVTX_UINTKEY64_H
#include "libcvtx.h"
/*============================================================================
UIntKey64.h

uint32 based key for working with 2D grids.

Copyright(c) 2019-2020 HJA Bird

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
============================================================================*/

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>

#ifdef __GNUC__
/* No need to include anything. */
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

/* A coordinate on a grid in 2D. */
class UIntKey64 {
public:
	union {
		struct {
			uint32_t x;
			uint32_t y;
		} k;
		uint64_t v;
	};

	UIntKey64() :v() {};
	inline UIntKey64(uint32_t x, uint32_t y) :k({ x, y }) {};
	inline UIntKey64(const UIntKey64& parent, uint32_t bit_shift, uint32_t lidx);

	/* Returns nearby grid points (including this one) in output buffer.
	Will truncate result to buffer_len keys. */
	void nearby_keys(
		int grid_radius,
		UIntKey64 *output_buffer, 
		size_t buffer_len);

	/* Number of nearby keys returned by the nearby keys function
	for a given grid radius. */
	static int num_nearby_keys(
		int grid_radius);

	/* Finds the nearest grid key to position based on the bottom
	corner of the axis system min. */
	static UIntKey64 nearest_key_min(
		bsv_V2f position, 
		float recip_grid_density, 
		bsv_V2f min);

	/* Finds the nearest grid key to position based on the centre
	of the grid system. */
	static UIntKey64 nearest_key_mean(
		bsv_V2f position,
		float recip_grid_density, 
		bsv_V2f mean);

	/* Returns the position represented by the grid key based on
	min coordinates and a grid density. */
	bsv_V2f to_position_min(
		float grid_density,
		bsv_V2f min);

	/* Returns the position represented by the grid key based on
	mean coordinates and a grid density. */
	bsv_V2f to_position_mean(
		float grid_density,
		bsv_V2f mean);

	/* Get a value in [0, 4] = x + 2*y  where x & y are 0 or 1
	according to the bit value at bit_idx. */
	const uint32_t lidx(uint32_t bit_idx);
	/* {x,y,z} in [0,1] to a linear index in [0,3]*/
	static uint32_t lidx(uint32_t xidx, uint32_t yidx);
	/* Linear index in[0,3] to {x,y} in [0,1] */
	static std::tuple<uint32_t, uint32_t> xyidx(uint32_t lidx);
	/* Get a value of the key's x idx at bit_idx */
	const uint32_t xidx(uint32_t bit_idx);
	/* Get a value of the key's y idx at bit_idx */
	const uint32_t yidx(uint32_t bit_idx);

	/* The extent to which keys match. Returns 32 for identical keys,
	0 for completely different keys. If they only match for the uppermost
	3 bytes, it'd return 3.*/
	const uint32_t matching_leading_bits(const UIntKey64& other);

	/* Converts a linear index in [0,7] to {x,y,z} in [0,1] and shifts it
	to a bit index. bit_idx == 0 results in setting lowest bits to xyz,
	31 results in setting upper most bits. */
	static UIntKey64 UIntKey64::partial_key(uint32_t lidx, uint32_t bit_idx);

	/* Returns as "{<xvalue>,<yvalue>,<zvalue>}" of key. */
	const std::string to_string();
};

inline bool operator==(const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v == rhs.v;
}
inline bool operator!=(const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v != rhs.v;
}
inline bool operator< (const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v < rhs.v;
}
inline bool operator> (const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v > rhs.v;
}
inline bool operator<=(const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v <= rhs.v;
}
inline bool operator>=(const UIntKey64& lhs, const UIntKey64& rhs) {
	return lhs.v >= rhs.v;
}

static inline UIntKey64 operator&(const UIntKey64& lhs, const UIntKey64& rhs) {
	UIntKey64 key;
	key.v = lhs.v & rhs.v;
	return key;
}
static inline UIntKey64 operator|(const UIntKey64& lhs, const UIntKey64& rhs) {
	UIntKey64 key;
	key.v = lhs.v | rhs.v;
	return key;
}
static inline UIntKey64 operator^(const UIntKey64& lhs, const UIntKey64& rhs) {
	UIntKey64 key;
	key.v = lhs.v ^ rhs.v;
	return key;
}
static inline UIntKey64 operator~(const UIntKey64& lhs) {
	UIntKey64 key;
	key.v = ~lhs.v;
	return key;
}
static inline UIntKey64 operator<<(const UIntKey64& lhs, const uint32_t shift) {
	UIntKey64 key;
	key.k.x = lhs.k.x << shift;
	key.k.y = lhs.k.y << shift;
	return key;
}
static inline UIntKey64 operator>>(const UIntKey64& lhs, const uint32_t shift) {
	UIntKey64 key;
	key.k.x = lhs.k.x >> shift;
	key.k.y = lhs.k.y >> shift;
	return key;
}


inline UIntKey64::UIntKey64(const UIntKey64& parent, uint32_t level, uint32_t idx)
{
	k.x = parent.k.x & (0xFFFFFFFF << (level + 1));
	k.y = parent.k.y & (0xFFFFFFFF << (level + 1));
	*this = *this | partial_key(idx, level);
}

inline uint32_t UIntKey64::lidx(uint32_t xidx, uint32_t yidx)
{
	assert(xidx < 2);
	assert(yidx < 2);
	return xidx + 2 * yidx;
}

inline std::tuple<uint32_t, uint32_t> UIntKey64::xyidx(uint32_t lidx) {
	assert(lidx < 4);
	uint32_t x = lidx & 0x1;
	uint32_t y = (lidx >> 1) & 0x1;
	return std::tie(x, y);
}

const inline uint32_t UIntKey64::lidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t x = (k.x & (0x1 << bit_idx)) > 0;
	uint32_t y = (k.y & (0x1 << bit_idx)) > 0;
	assert(x < 2);
	assert(y < 2);
	return x + 2 * y;
}

const inline uint32_t UIntKey64::xidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t x = (k.x & 0x1) << bit_idx;
	return x;
}
const inline uint32_t UIntKey64::yidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t y = (k.y & 0x1) << bit_idx;
	return y;
}

const inline uint32_t UIntKey64::matching_leading_bits(const UIntKey64& key) {
	uint32_t res, x, y;
	UIntKey64 tmp = key ^ *this;
#ifdef __GNUC__
	assert(false); /* TO DO. */
	x = __builtin_clz((unsigned long)tmp.k.x);
	y = __builtin_clz((unsigned long)tmp.k.y);
	x = !key.k.x ? 32 - x : 32;
	y = !key.k.y ? 32 - y : 32;
#elif defined(_MSC_VER)
	unsigned char nx, ny;
	nx = _BitScanReverse((unsigned long*)&x, (unsigned long)tmp.k.x);
	ny = _BitScanReverse((unsigned long*)&y, (unsigned long)tmp.k.y);
	res = 0;
	if (!nx && !ny) { return 32; }
	x = nx ? 31 - x : 32;
	y = ny ? 31 - y : 32;
#endif
	res = x < y ? x : y;
	return res;
}

inline UIntKey64 UIntKey64::partial_key(uint32_t lidx, uint32_t bit_idx)
{
	assert(bit_idx < 32);
	UIntKey64 k;
	std::tie(k.k.x, k.k.y) = xyidx(lidx);
	k = k << bit_idx;
	return k;
}

void sort_perm_UIntKey64(
	UIntKey64* gridkeys,
	unsigned int* key_start, size_t num_items);



#endif /* CVTX_UINTKEY64_H */
