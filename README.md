# cvortex

cvortex is a C library for accelerating vortex particle methods in unsteady aerodynamics with an 
emphisis on ease of use. Currently, computations are best kept to the region of 100,000 particles.

The library is accelerated using OpenMP and OpenCL 1.2.
It can be run on AMD, Intel and Nvidia gpus.

There are several ways to use the library:
- The Julia language wrapper is the easiest - [CVortex.jl](https://github.com/hjabird/CVortex.jl),
although it doesn't yet expose the whole API and comes with a small overhead.
- Prebuilt binaries are available in the release section. Currently these 
are built for x86-64 compatable CPUs on Windows (MSVC) and Linux (Ubuntu GCC7).
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

### The Basics

The library supports 3 boundary elements:
 - The 3D vortex particle `cvtx_P3D`
 - The 3D straight, singular vortex filament `cvtx_F3D`
 - The 2D vortex particle `cvtx_P2D`
 
The vortex particle can be regularised using `cvtx_VortFunc`.

When using the library, you'll want to use different functions according to the number of interactions:
 - Single to single, `S2S`.
 - Many to single, `M2S`.
 - Many to many, `M2M`.
 
Additionally, different types of interaction can occur:
 - Most often, the induced velocity at point, `vel`.
 - With 3D vortex particles, the vortex stretching term leading to a change in vorticity, `dvort`.
 - And viscocity (for some regularisations). `visc_dvort`.
 
The library includes apparatus to model the effect of vortex filaments on vortex particles in 3D,
but since the vortex filaments are singular, this is restricted to invicid interaction.

Consequently, a function name is generally of the form `cvtx_OOO_III_FN` where
`OOO` is the object type (`P3D`, `F3D` or `P2D`), `III` is according to the number
of interactions (ie `S2S`, `M2S` or `M2M`) and `FN` is what is being computed (`vel`, `dvort` 
or `visc_dvort`).

For example, to find the velocity imposed on a set of points by a set of 3D vortex particles, 
one would use `cvtx_P3D_M2M_vel`.

### Regularisation

Regularised vortex particles are far more useful than singular ones. Consequently, 
the library included regularisation functions, where functions are packaged together
into a struct called of type `cvtx_VortFunc`. The library includes functions to generate
several popular regularisations in both 2D and 3D:
 - No regularisation - singular: `cvtx_VortFunc_singular`.
 - Winckelmans' high order algebraic regularisation `cvtx_VortFunc_winckelmans`.
 - Gaussian regularisation `cvtx_VortFunc_gaussian`.
 - Planetary regularisation `cvtx_VortFunc_planetary`.
The structures do not contain the regularisation distance - this is fed into 
functions as an argument which is ignored for singular regularisation. Not all
regularisations support viscous interaction via `visc_dvort`.

### Function arguments

Generally, the best place to see the available functions is `libcvtx.h` - the 
public interface header. Displayed here are the function signitures available.

Of note is the method by which arrays of filaments or vortex particles are
passed into the functions. Lets look at one function in particular:
```
void cvtx_F3D_M2M_dvort(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array);
```
This function computes the vortex stretching term induced on a set of 3D vortex
particles by a set of vortex filaments. Both the filaments and vortex
particles are passed into the function by something of the form
`**objects`. This is an array of points to vortex particles or vortex filments.
ie. `*(objects[0])` should give the object.

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
- [ExaFMM-t](https://github.com/exafmm/exafmm-t):  A fast multipole accelerated code utilising CUDA, C++
and MPI on Linux. Big, ambititious and fast (I'm assuming). Also [PyExaFMM](https://github.com/exafmm/pyexafmm) (incomplete?).
- [PETFMM](https://bitbucket.org/petfmm/):  Superseeded by ExaFMM.
- [KIFMM](https://github.com/jeewhanchoi/kifmm--hybrid--double-only):  A fast multipole CUDA/CPU hybrid
code demonstrating kernal independence.
- [Bonsai](https://github.com/treecode/Bonsai) Barns-Hut gravity code.
- [ChaNGa](https://github.com/N-BodyShop/changa/wiki/ChaNGa) Another Barns-Hut gravity code. But this time with lots of documentation. Appears to support MPI on Linux.
- FLOWVPM: A closed source vortex particle code. [Edo Alvarez's site](https://edoalvarezr.github.io/projects/01-aerodynamics.html).

## Authors
HJA Bird

## Licence 
You get your copy under the MIT licence. 






