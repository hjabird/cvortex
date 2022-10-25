#ifndef CVTX_UINTKEY96_H
#define CVTX_UINTKEY96_H
#include "libcvtx.h"
/*============================================================================
UIntKey96.c

uint32 based key for working with 3D grids.

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

#include <cstdint>
#include <tuple>
#include <string>

#ifdef __GNUC__
/* No need to include anything. */
#elif defined(_MSC_VER)
#include <intrin.h>
#endif

/* A coordinate on a grid in 3D. */
class UIntKey96 {
public:
	/* DATA */
	union {
		struct {
			uint32_t x;
			uint32_t y;
			uint32_t z;
		} k;
		struct {
			uint64_t lo;
			uint32_t up;
		} v;
		/* Note uint32 lo, uint64 up isn't possible due to allignment. */
	};

	UIntKey96() :v() {};
    inline UIntKey96(uint32_t x, uint32_t y, uint32_t z) {
        k.x = x; k.y = y; k.z = z;
    };
	inline UIntKey96(const UIntKey96& parent, uint32_t bit_shift, uint32_t lidx);

	/* Returns nearby grid points (including this one) in output buffer.
	Will truncate result to buffer_len keys. */
	void nearby_keys(
		int grid_radius,
		UIntKey96* output_buffer,
		size_t buffer_len);

	/* Number of nearby keys returned by the nearby keys function
	for a given grid radius. */
	static size_t num_nearby_keys(
		int grid_radius);

	/* Finds the nearest grid key to position based on the bottom 
	corner of the axis system min. */
	static UIntKey96 nearest_key_min(
		bsv_V3f position,
		float recip_grid_density,
		bsv_V3f min);

	/* Finds the nearest grid key to position based on the centre
	of the grid system. */
	static UIntKey96 nearest_key_mean(
		bsv_V3f position,
		float recip_grid_density, 
		bsv_V3f mean);

	/* Returns the position represented by the grid key based on
	min coordinates and a grid density. */
	bsv_V3f to_position_min(
		float grid_density,
		bsv_V3f min);

	/* Returns the position represented by the grid key based on
	mean coordinates and a grid density. */
	bsv_V3f to_position_mean(
		float grid_density,
		bsv_V3f mean);

	/* Get a value in [0, 7] = x + 2*y + 4*z where x, y & z are 0 or 1
	according to the bit value at bit_idx. */
	const uint32_t lidx(uint32_t bit_idx);
	/* {x,y,z} in [0,1] to a linear index in [0,7]*/
	static uint32_t lidx(uint32_t xidx, uint32_t yidx, uint32_t zidx);
	/* Linear index in[0,7] to {x,y,z} in [0,1] */
	static std::tuple<uint32_t, uint32_t, uint32_t> xyzidx(uint32_t lidx);
	/* Get a value of the key's x idx at bit_idx */
	const uint32_t xidx(uint32_t bit_idx);
	/* Get a value of the key's y idx at bit_idx */
	const uint32_t yidx(uint32_t bit_idx);
	/* Get a value of the key's z idx at bit_idx */
	const uint32_t zidx(uint32_t bit_idx);

	/* The extent to which keys match. Returns 32 for identical keys,
	0 for completely different keys. If they only match for the uppermost
	3 bytes, it'd return 3.*/
	const uint32_t matching_leading_bits(const UIntKey96 &other);

	/* Converts a linear index in [0,7] to {x,y,z} in [0,1] and shifts it
	to a bit index. bit_idx == 0 results in setting lowest bits to xyz,
	31 results in setting upper most bits. */
	static UIntKey96 partial_key(uint32_t lidx, uint32_t bit_idx);

	/* Returns as "{<xvalue>,<yvalue>,<zvalue>}" of key. */
	const std::string to_string();
};

static inline bool operator==(const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo == rhs.v.lo) && (lhs.v.up == rhs.v.up); 
}
static inline bool operator!=(const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo != rhs.v.lo) && (lhs.v.up != rhs.v.up);
}
static inline bool operator< (const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo < rhs.v.lo) && (lhs.v.up < rhs.v.up);
}
static inline bool operator> (const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo > rhs.v.lo) && (lhs.v.up > rhs.v.up);
}
static inline bool operator<=(const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo <= rhs.v.lo) && (lhs.v.up <= rhs.v.up);
}
static inline bool operator>=(const UIntKey96& lhs, const UIntKey96& rhs) {
	return (lhs.v.lo >= rhs.v.lo) && (lhs.v.up >= rhs.v.up);
}
static inline UIntKey96 operator&(const UIntKey96& lhs, const UIntKey96& rhs) {
	UIntKey96 key;
	key.v.lo = lhs.v.lo & rhs.v.lo;
	key.v.up = lhs.v.up & rhs.v.up;
	return key;
}
static inline UIntKey96 operator|(const UIntKey96& lhs, const UIntKey96& rhs) {
	UIntKey96 key;
	key.v.lo = lhs.v.lo | rhs.v.lo;
	key.v.up = lhs.v.up | rhs.v.up;
	return key;
}
static inline UIntKey96 operator^(const UIntKey96& lhs, const UIntKey96& rhs) {
	UIntKey96 key;
	key.v.lo = lhs.v.lo ^ rhs.v.lo;
	key.v.up = lhs.v.up ^ rhs.v.up;
	return key;
}
static inline UIntKey96 operator~(const UIntKey96& lhs) {
	UIntKey96 key;
	key.v.lo = ~lhs.v.lo;
	key.v.up = ~lhs.v.up;
	return key;
}
static inline UIntKey96 operator<<(const UIntKey96& lhs, const uint32_t shift) {
	UIntKey96 key;
	key.k.x = lhs.k.x << shift;
	key.k.y = lhs.k.y << shift;
	key.k.z = lhs.k.z << shift;
	return key;
}
static inline UIntKey96 operator>>(const UIntKey96& lhs, const uint32_t shift) {
	UIntKey96 key;
	key.k.x = lhs.k.x >> shift;
	key.k.y = lhs.k.y >> shift;
	key.k.z = lhs.k.z >> shift;
	return key;
}

inline UIntKey96::UIntKey96(const UIntKey96& parent, uint32_t level, uint32_t idx)
{
	k.x = parent.k.x & (0xFFFFFFFF << (level + 1));
	k.y = parent.k.y & (0xFFFFFFFF << (level + 1));
	k.z = parent.k.z & (0xFFFFFFFF << (level + 1));
	*this = *this | partial_key(idx, level);
}

inline uint32_t UIntKey96::lidx(uint32_t xidx, uint32_t yidx, uint32_t zidx)
{
	assert(xidx < 2);
	assert(yidx < 2);
	assert(zidx < 2);
	return xidx + 2 * yidx + 4 * zidx;
}

inline std::tuple<uint32_t, uint32_t, uint32_t> UIntKey96::xyzidx(uint32_t lidx) {
	assert(lidx < 8);
	uint32_t x = lidx & 0x1;
	uint32_t y = (lidx >> 1) & 0x1;
	uint32_t z = (lidx >> 2) & 0x1;
	return std::tie(x, y, z);
}

const inline uint32_t UIntKey96::lidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t x = (k.x & (0x1 << bit_idx)) > 0;
	uint32_t y = (k.y & (0x1 << bit_idx)) > 0;
	uint32_t z = (k.z & (0x1 << bit_idx)) > 0;
	assert(x < 2);
	assert(y < 2);
	assert(z < 2);
	return x + 2 * y + 4 * z;
}

const inline uint32_t UIntKey96::xidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t x = (k.x & 0x1) << bit_idx;
	return x;
}
const inline uint32_t UIntKey96::yidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t y = (k.y & 0x1) << bit_idx;
	return y;
}
const inline uint32_t UIntKey96::zidx(uint32_t bit_idx)
{
	assert(bit_idx < 32);
	uint32_t z = (k.z & 0x1) << bit_idx;
	return z;
}

const inline uint32_t UIntKey96::matching_leading_bits(const UIntKey96& key) {
	uint32_t res, x, y, z;
	UIntKey96 tmp = key ^ *this;
#ifdef __GNUC__
	assert(false); /* TO DO. */
	x = __builtin_clz((unsigned long)tmp.k.x);
	y = __builtin_clz((unsigned long)tmp.k.y);
	z = __builtin_clz((unsigned long)tmp.k.z);
	x = !key.k.x ? 32 - x : 32;
	y = !key.k.y ? 32 - y : 32;
	z = !key.k.z ? 32 - z : 32;
#elif defined(_MSC_VER)
	unsigned char nx, ny, nz;
	nx = _BitScanReverse((unsigned long*)&x, (unsigned long)tmp.k.x);
	ny = _BitScanReverse((unsigned long*)&y, (unsigned long)tmp.k.y);
	nz = _BitScanReverse((unsigned long*)&z, (unsigned long)tmp.k.z);
	res = 0;
	if (!nx && !ny && !nz) { return 32; }
	x = nx ? 31-x : 32;
	y = ny ? 31-y : 32;
	z = nz ? 31-z : 32;
#endif
	res = x < y ? x : y;
	res = res < z ? res : z;
	return res;
}

inline UIntKey96 UIntKey96::partial_key(uint32_t lidx, uint32_t bit_idx)
{
	assert(bit_idx < 32);
	UIntKey96 k; 
	std::tie(k.k.x, k.k.y, k.k.z) = xyzidx(lidx);
	k = k << bit_idx;
	return k;
}

void sort_perm_UIntKey96(
	UIntKey96 *gridkeys,
	unsigned int* key_start, size_t num_items);


#endif /* CVTX_UINTKEY96_H */
