This folder contains files with methods from `libcvtx.h`
- `F3D.c`: 3D vortex filaments methods (CPU + calls to GPU methods). 
- `P3D.c`: 3D vortex particle methods (CPU + calls to GPU methods). 
- `P2D.c`: 2D vortex particle methods (CPU + calls to GPU methods).
- `VortFunc.c`: Vortex regularisation functions.
- `accelerators.c`: Handeling of accelerator API.

If compiled with `CVTX_USING_OPENCL`the following files are also used:
- `nbody.cl`: The opencl implementation of many to many interactions. This is embedded as text within the final library, hence is written as a C string.
- `ocl_XXX.h/c`: Host side opencl implementation of 3D/2D vortex particle/filament methods.
- `opencl_acc.h/c`: Apparatus for handeling devices and building the OpenCL programs.
