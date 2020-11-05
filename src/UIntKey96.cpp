#include "UIntKey96.h"
/*============================================================================
UIntKey96.cpp

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

#include <cassert>
#include <cmath>
#include "array_methods.h"


/*	DEFINITIONS ------------------------------------------------------------*/

void sort_perm_UIntKey96(
	UIntKey96 *gridkeys,
	unsigned int* key_start, size_t num_items) {
	assert(num_items >= 0);
	if (num_items == 0) {
		return;	/* Nothing to do. */
	}
	else {
		sort_perm_multibyte_radix8(
			(unsigned char *)gridkeys, sizeof(UIntKey96),
			key_start, num_items);
	}
}

void UIntKey96::nearby_keys(int grid_radius, UIntKey96* output_buffer, size_t buffer_len)
{
	int ii, jj, kk; /* 3D indices. */
	uint32_t o_idx = 0; /* Output to be kept less than buffer len. */
	uint32_t kx = k.x, ky = k.y, kz = k.z;
	assert(kx - grid_radius < kx); /* Grid indexes must all be +ve. */
	assert(ky - grid_radius < ky);
	assert(kz - grid_radius < kz);

	for (ii = -grid_radius; ii <= grid_radius; ++ii) {
		for (jj = -grid_radius; jj <= grid_radius; ++jj) {
			for (kk = -grid_radius; kk <= grid_radius; ++kk) {
				size_t idx = 
					(ii + grid_radius) * (2*grid_radius + 1) * (2*grid_radius + 1) +
					(jj + grid_radius) * (2*grid_radius + 1) +
					(kk + grid_radius);
				if (idx < buffer_len) {
					output_buffer[idx] = UIntKey96(kx + ii, ky + jj, kz + kk);
				}
			}
		}
	}
	return;
}

size_t UIntKey96::num_nearby_keys(int grid_radius)
{
	assert(grid_radius >= 0);
	size_t nk = (2 * grid_radius + 1) * (2 * grid_radius + 1) * (2 * grid_radius + 1);
	return nk;
}

const std::string UIntKey96::to_string()
{
	std::string str;
	str = "{" + std::to_string(k.x) + ", " + std::to_string(k.y) + ", " + std::to_string(k.z) + "}";
	return str;
}

UIntKey96 UIntKey96::nearest_key_min(
	bsv_V3f position, 
	float recip_grid_density, 
	bsv_V3f min)
{
	assert(recip_grid_density > 0.f);
	UIntKey96 ret;
	float x, y, z;
	x = position.x[0];
	y = position.x[1];
	z = position.x[2];
	x = (x - min.x[0]) * recip_grid_density;
	y = (y - min.x[1]) * recip_grid_density;
	z = (z - min.x[2]) * recip_grid_density;
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	ret.k.x = (unsigned int)(roundf(x));
	ret.k.y = (unsigned int)(roundf(y));
	ret.k.z = (unsigned int)(roundf(z));
	return ret;
}

UIntKey96 UIntKey96::nearest_key_mean(
	bsv_V3f position,
	float recip_grid_density,
	bsv_V3f mean)
{
	assert(recip_grid_density > 0.f);
	UIntKey96 ret;
	float x, y, z;
	x = position.x[0];
	y = position.x[1];
	z = position.x[2];
	x = (x - mean.x[0]) * recip_grid_density;
	y = (y - mean.x[1]) * recip_grid_density;
	z = (z - mean.x[2]) * recip_grid_density;
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	ret.k.x = (uint32_t)(round(x) + 2147483647.5); /* uint32_max / 2*/
	ret.k.y = (uint32_t)(round(y) + 2147483647.5);
	ret.k.z = (uint32_t)(round(z) + 2147483647.5);
	return ret;
}

bsv_V3f UIntKey96::to_position_min(float grid_density, bsv_V3f min)
{
	bsv_V3f pos = min;
	pos.x[0] += grid_density * k.x;
	pos.x[1] += grid_density * k.y;
	pos.x[2] += grid_density * k.z;
	return pos;
}

bsv_V3f UIntKey96::to_position_mean(float grid_density, bsv_V3f mean)
{
	bsv_V3f pos = mean;
	pos.x[0] += (float) (grid_density * ((double)k.x - 2147483647.5));
	pos.x[1] += (float) (grid_density * ((double)k.y - 2147483647.5));
	pos.x[2] += (float) (grid_density * ((double)k.z - 2147483647.5));
	return pos;
}
