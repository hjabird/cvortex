/*============================================================================
nbody.cl

OpenCL implementation of vortex particle N-Body problem using brute force.

Copyright(c) 2018 HJA Bird

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
#define CVTX_WORKGROUP_SIZE 128

/*----------------------------------------------------------------------------
Definitions for the repeated body of kernels
----------------------------------------------------------------------------*/

#define CVTX_P_INDVEL_START 	float3 rad, num, ret;				\
	float cor, den, rho, g;											\
	__local double3 reduction_workspace[CVTX_WORKGROUP_SIZE];		\
	/* Particle idx, mes_pnt idx and local work item idx */			\
	uint pidx, midx, widx, loop_idx;								\
	pidx = get_global_id(0);										\
	midx = get_global_id(1);										\	
	widx = get_local_id(0);											\
	if(isequal(particle_locs[pidx], mes_locs[midx])){				\
		ret.x = 0.0;												\
		ret.y = 0.0;												\
		ret.z = 0.0;												\
	}																\
	else 															\
	{																\
		rad = mes_locs[midx] - particle_locs[pidx];					\
		rho = length(rad) / particle_rads[pidx];				
		/* Fill in g calc here */
#define CVTX_P_INDVEL_END 											\
		cor = - g / (4 * acos(-1));									\
		den = powr(length(rad), 3);									\
		num = cross(rad, particle_vorts[pidx]);						\
		ret = num * (cor / den);									\
	}																\
	reduction_workspace[widx] = ret;								\
	/* Now sum to a single value. */								\
	for(loop_idx = 2; loop_idx < CVTX_WORKGROUP_SIZE; 				\
		loop_idx *= 2){												\
		mem_barrier();												\
		if( widx < CVTX_WORKGROUP_SIZE/loop_idx ){					\
			reduction_workspace[widx] = reduction_workspace[widx] 	\ 
				+ reduction_workspace[widx * 2];					\
		}															\
	}																\
	results[midx] = reduction_workspace[0];							\
	return;															
	
#define CVTX_P_IND_DVORT_START										\
	float3 ret, rad, cross_om, t2, t21, t21n, t22, t224;			\
	float g, f, radd, rho, t1, t21d, t221, t222, t223;				\
	/* self (inducing particle) index, induced particle index */	\
	uint sidx, indidx, widx, loop_idx;								\
	sidx = get_global_id(0);										\
	indidx = get_global_id(1);										\	
	widx = get_local_id(0);											\
	if(isequal(particle_locs[sidx], mes_locs[indidx])){				\
		ret.x = 0.0;												\
		ret.y = 0.0;												\
		ret.z = 0.0;												\
	}																\
	else 															\
	{																\
		rad = induced_locs[indidx], particle_locs[sidx];			\
		radd = length(rad);											\
		rho = radd / particle_rads[sidx];							
		/* FILL in f & g calc here! */
#define CVTX_P_IND_DVORT_END										\		
		cross_om = cross(induced_vorts[indidx], 					\
			particle_vorts[sidx]);									\
		t1 = 1. / (4. * acos(-1) * powr(particle_rads[sidx], 3));	\
		t21n = cross_om * -g;										\
		t21d = rho * rho * rho;										\
		t21 = t21n / t21d;											\
		t221 = 1. / (radd * radd);									\
		t222 = (3 * g) / (rho * rho * rho) - f;						\
		t223 = dot(particle_vorts[sidx], rad);						\
		t224 = cros(rad, particle_vorts[sidx]);						\
		t22 = t224 * t221 * t222 * t223;							\
		t2 = t21 + t22;												\
		ret = t2 * t1;												\
	}	reduction_workspace[widx] = ret;							\
	/* Now sum to a single value. */								\
	for(loop_idx = 2; loop_idx < CVTX_WORKGROUP_SIZE; 				\
		loop_idx *= 2){												\
		mem_barrier();												\
		if( widx < CVTX_WORKGROUP_SIZE/loop_idx ){					\
			reduction_workspace[widx] = reduction_workspace[widx] 	\ 
				+ reduction_workspace[widx * 2];					\
		}															\
	}																\
	results[midx] = reduction_workspace[0];							\
	return;															
	
__kernel void cvtx_nb_Particle_ind_vel_singular(
	__global float3* particle_locs,
	__global float3* particle_vorts,
	__global float* particle_rads,
	__global float3* mes_locs,
	__global double3* results)
{
	CVTX_P_INDVEL_START
	g = 1.;
	CVTX_P_INDVEL_END
}

__kernel void cvtx_nb_Particle_ind_dvort_singular(
	__global float3* particle_locs,
	__global float3* particle_vorts,
	__global float* particle_rads,
	__global float3* induced_locs,
	__global float3* induced_vorts,
	__global double3* results)
{
	CVTX_P_IND_DVORT_START
		g = 1.;
		f = 0.;
	CVTX_P_IND_DVORT_END
}
