# cvortex

cvortex is a C library for accelerating vortex particle methods in unsteady aerodynamics with an 
emphisis on ease of use. Currently, computations are best kept to the region of 100,000 particles.

The library is accelerated using OpenMP and OpenCL 1.2, so AMD,
Intel and Nvidia gpus are capable of accelerating computation.

Want to avoid the pain of building and using C? Try:
- The prebuilt binaries available in the release section.
- The Julia language wrapper - [cvortex.jl](https://github.com/hjabird/cvortex.jl)
- Consider wrapping cvortex for another language - its not a 
complicated interface and precompiled binaries are available.


## Building

It should be possible to build cvortex on both Windows and Linux (and Mac, but this
is untested).

First, download my branch of vcpkg:
```
git clone https://github.com/hjabird/vcpkg
```
Next, bootstrap it :`./bootstrap-vcpkg.bat` or `./bootstrap-vcpkg.sh` depending on platform. 
Download and build cvortex's dependencies:
```
./vcpkg install bsv opencl --triplet x64-windows
```
My experience with installing opencl using vcpkg on Ubuntu has been variable. You may
have to work this out yourself it it fails. Where vcpkg fails, normal ubuntu
package management may work anyway. Try `sudo apt-get install ocl-icd-opencl-dev`.

Now we can download and build cvortex.
```
git clone https://github.com/hjabird/cvortex
cd cvortex
mkdir build
cd build
```

### Windows
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Path/To/vcpkg/scripts/buildsystems/vcpkg.cmake -G"Visual Studio 15 2017 Win64"
```
And then use Visual Studio to build cvortex.
### Ubuntu
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/Path/To/vcpkg/scripts/buildsystems/vcpkg.cmake
make
```
To build a debug version go
```
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

## Usage

### Including and initialising

Only one header need to be included - `libcvortex.h`.
```
#include <cvortex/libcvortex.h>
```
You MUST initialise the cvortex library:
```
cvtx_initialise();
```
Likewise, when you're finished, call
```
cvtx_finalise();
```
You'll notice that all calls to cvortex start with `cvtx`.

### Vortex particles

Vortex particles are defined as
```
typedef struct {
	bsv_V3f coord;
	bsv_V3f vorticity;
	float volume;
} cvtx_Particle;
```
where `volume` only matters if you're using particle strength exchange to model viscocity.
This feature might be removed in a later release.

You'll be interested in the following functions:
```
/* Single particle - single induced */
cvtx_Particle_ind_vel
cvtx_Particle_ind_dvort
cvtx_Particle_visc_ind_dvort
/* Multiple particle - single induced */
cvtx_ParticleArr_ind_vel
cvtx_ParticleArr_ind_dvort
cvtx_ParticleArr_visc_ind_dvort
/* Multiple particle - multiple induced */
cvtx_ParticleArr_Arr_ind_vel
cvtx_ParticleArr_Arr_ind_dvort
cvtx_ParticleArr_Arr_visc_ind_dvort
```
Multiple to single is accelerated with OpenMP.
Multiple-Multiple is accelerated using OpenCL.

You'll need to use regularisation functions as inputs to the above.
```
typedef struct {
	float(*g_fn)(float rho);
	float(*zeta_fn)(float rho);
	void(*combined_fn)(float rho, float* g, float* zeta);
	float(*eta_fn)(float rho);
	char cl_kernel_name_ext[32];
} cvtx_VortFunc;

CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_singular(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_winckelmans(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_planetary(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_gaussian(void);
```
You can create your own regularisation function, but it won't be GPU accelerated
(although it will be accelerated using OpenMP.

You'll have to do your own ODE solving - this isn't included 
in the library. Whilst forward-Euler schemes are easy, its generally believed
that high order schemes are needed for better accuracy.

### Vortex Filaments

All vortex filaments are straight and singular here. They
are represented with a start and an end.
```
typedef struct {
	bsv_V3f start, end;		/* Beginning & end coordinates of line segment */
	float strength;			/* Vort per unit length */
} cvtx_StraightVortFil;
```

The API is similar to that of the vortex particle, but all interaction is viscous
and, consequently, viscosity cannot be modelled.
```
CVTX_EXPORT bsv_V3f cvtx_StraightVortFil_ind_vel;
CVTX_EXPORT bsv_V3f cvtx_StraightVortFil_ind_dvort;

CVTX_EXPORT bsv_V3f cvtx_StraightVortFilArr_ind_vel;
CVTX_EXPORT bsv_V3f cvtx_StraightVortFilArr_ind_dvort;

CVTX_EXPORT void cvtx_StraightVortFilArr_Arr_ind_vel;
CVTX_EXPORT void cvtx_StraightVortFilArr_Arr_ind_dvort(
```

Additionally, there is a function to create an influence matrix for use
by vortex lattice solvers, etc. This is called as
```
CVTX_EXPORT void cvtx_StraightVortFilArr_inf_mtrx;
```
Note that the matrix quickly gets big - don't include too many vortex
filaments or you'll have issues.

### Accelerators
You'll want a way to control the accelerators on your platform. Right now, 
cvortex will only look for GPUs. If it can't find any it'll use its multithreaded
CPU implementation for everything. If you have multiple possible GPUs, it'll
use the first one it finds. You can enable multiple accelerators, but right now
it doesn't do anything useful. You can control all this using the accelerator API.

```
CVTX_EXPORT int cvtx_num_accelerators();
CVTX_EXPORT int cvtx_num_enabled_accelerators();
CVTX_EXPORT char* cvtx_accelerator_name(int accelerator_id);
CVTX_EXPORT int cvtx_accelerator_enabled(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_enable(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_disable(int accelerator_id);
```

`cvtx_num_accelerators()` gives you the number of GPUs found on the platform.
`cvtx_num_enabled_accelerators()` gives you the number that are set to be used. If
this number is zero, then the multithreaded CPU implementation is used.

Details about the accelerators can be accessed by index. A buffer containing
the name of the first accelerator can be accessed by using
```
char *name = cvtx_accelerator_name(0);
```
This name buffer is internally managed - don't release it.

You can find out if its enabled using `cvtx_accelerator_enabled`. 1 indicates yes, 0 no.

You can enable and disable accelerators. You might to do this to ensure that only your
fastest GPU is used (for instance, if you have a integrated GPU and a discrete GPU
and don't want to use the integrated one), or to disable all your GPUs such that 
only the CPU implementation is used.

## Performance
TO DO.

Currently cvortex only uses naive algorithms, so the n body problem scales as n<sup>2</sup>.
To obtain best performance, try and use as few calls as possible. If there aren't enough
input measurement points or particles, the CPU implementation is used. Also, note that
for implementation reasons, particles are internally grouped into sets of 256. Hence
Modelling 512 and 700 particles will consume the same abount of time for a given 
number of measurement points.

## Alternative libaries
A lack of easy to use, cross platform and non-CUDA alternatives is why this library was written. 
However, you may be interested in the following:
- [EXAFMM](https://github.com/exafmm/exafmm):  A fast multipole accelerated code utilising CUDA, C++
and MPI on Linux. Big, ambititious and fast (I'm assuming).
- [PETFMM](https://bitbucket.org/petfmm/):  Superseeded by ExaFMM.
- [KIFMM](https://github.com/jeewhanchoi/kifmm--hybrid--double-only):  A fast multipole CUDA/CPU hybrid
code demonstrating kernal independence.
- [FLOWVLM](https://github.com/byuflowlab/FLOWVLM):  A Julia implementation of vortex lattice 
and vortex particle methods according to images? Impressive looking simulations of wind turbines are 
visible on [Edo Alvarez's site](https://edoalvarezr.github.io/projects/01-aerodynamics.html).

## Authors
HJA Bird

## Licence 
You get your copy under the MIT licence. 






