#include "UIntKey64.hpp"
/*============================================================================
UIntKey64.c

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
#include <string>
#include "array_methods.h"

/*	DEFINITIONS ------------------------------------------------------------*/

void sort_perm_UIntKey64(
	UIntKey64* gridkeys,
	unsigned int* key_start, size_t num_items) {
	if (num_items == 0) {
		return;	/* Nothing to do. */
	}
	else {
		sort_perm_multibyte_radix8(
			(unsigned char*)gridkeys, sizeof(UIntKey64),
			key_start, num_items);
	}
}

void UIntKey64::nearby_keys(int grid_radius, UIntKey64* output_buffer, size_t buffer_len)
{
	int i, j;
	uint32_t kx = k.x, ky = k.y;
	assert(kx - grid_radius < kx);
	assert(ky - grid_radius < ky);
	for (i = -grid_radius; i <= grid_radius; ++i) {
		for (j = -grid_radius; j <= grid_radius; ++j) {
			size_t idx = (i + grid_radius) * (2 * grid_radius + 1) + (j + grid_radius);
			if (idx < buffer_len) {
				output_buffer[idx] = UIntKey64(kx + i, ky + j);
			}
		}
	}
	return;
}

int UIntKey64::num_nearby_keys(int grid_radius)
{
	assert(grid_radius >= 0);
	int nk = (2 * grid_radius + 1) * (2 * grid_radius + 1);
	return nk;
}

const std::string UIntKey64::to_string()
{
	std::string str;
	str = "{" + std::to_string(k.x) + ", " + std::to_string(k.y) + "}";
	return str;
}

UIntKey64 UIntKey64::nearest_key_min(
	bsv_V2f position,
	float recip_grid_density,
	bsv_V2f min)
{
	assert(recip_grid_density > 0.f);
	UIntKey64 ret;
	double x, y; /* Otherwise we definitely can't use full 32 bits of precision per dim. */
	x = (double) position.x[0];
	y = (double) position.x[1];
	x = (x - min.x[0]) *recip_grid_density;
	y = (y - min.x[1]) * recip_grid_density;
	assert(x >= 0);
	assert(y >= 0);
	ret.k.x = (unsigned int)(roundf(x));
	ret.k.y = (unsigned int)(roundf(y));
	return ret;
}

UIntKey64 UIntKey64::nearest_key_mean(
	bsv_V2f position,
	float recip_grid_density,
	bsv_V2f mean)
{
	UIntKey64 ret;
	double x, y; /* Otherwise we definitely can't use full 32 bits of precision per dim. */
	x = (double) position.x[0];
	y = (double) position.x[1];
	x = (x - mean.x[0]) * recip_grid_density;
	y = (y - mean.x[1]) * recip_grid_density;
	assert(x >= 0);
	assert(y >= 0);
	ret.k.x = (uint32_t)(round(x) + 2147483647.5); /* uint32_max / 2*/
	ret.k.y = (uint32_t)(round(y) + 2147483647.5);
	return ret;
}

bsv_V2f UIntKey64::to_position_min(float grid_density, bsv_V2f min)
{
	bsv_V2f pos = min;
	pos.x[0] += grid_density * k.x;
	pos.x[1] += grid_density * k.y;
	return pos;
}

bsv_V2f UIntKey64::to_position_mean(float grid_density, bsv_V2f mean)
{
	bsv_V2f pos = mean;
	pos.x[0] += (float) (grid_density * ((double)k.x - 2147483647.5));
	pos.x[1] += (float) (grid_density * ((double)k.y - 2147483647.5));
	return pos;
}
