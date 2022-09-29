#ifndef CVTX_GRIDPARTICLEQUADTREE_H
#define CVTX_GRIDPARTICLEQUADTREE_H
/*============================================================================
GridParticleQuadtree.h

A set of vortex particles on a grid represented by a sparse quadtree.

NB: Almost identical to GridParticleOcttree - features / bugs can be 
mirrored. 

Copyright(c) 2020 HJA Bird

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

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include <bsv/bsv_V2f.h>

#include "UIntKey64.hpp"

class GridParticleQuadtree;
class GridParticleQuadtreeBranch;

typedef struct {
	uint32_t type;	/* 0 = unallocated, 1 == branch, 2 = float (strenght) */
	union {
		float f;
		uint32_t u;
	} v;
} cvalue_t;

class GridParticleQuadtreeBranch { /* Used only by GridParticleQuadtree */
public:
	GridParticleQuadtreeBranch();
	GridParticleQuadtreeBranch(int level, UIntKey64 key);
protected:
	int m_level;
	UIntKey64 m_key;	/* Partial key according to m_level */
	std::array<cvalue_t, 4> m_child_idxs;
	friend class GridParticleQuadtree;
};


class GridParticleQuadtree {
public:
	/* Adds vorticity str at location given by key.
	It is added to any vorticity already at that grid point.*/
	void add_particle(UIntKey64 key, float str);
	void add_particles(
		std::vector<UIntKey64> key, std::vector<float> str);

	/* The number of particles within the tree. */
	size_t number_of_particles();

	/* Get a load of index / strength pairs is a long list.
	Main method to get particles out of the tree. */
	void flatten_tree(UIntKey64* idxs, float* strs, int max_particles);

	/* Merge another GridParticleOctree into this one. */
	void merge_in(const GridParticleQuadtree&);

	void clear();

	GridParticleQuadtree();

protected:
	/* Branches refer to their children by index, deindexed in this object. */
	std::vector<GridParticleQuadtreeBranch> m_branches;
	/* Vorticities are held by the branch object. But we keep count. */
	uint64_t m_vortex_count;
};


/*
HOW IS INDEXING IMPLEMENTED HERE?

UIntKey64 has 2 uint32_t subkeys.
The biggest bit of these can be found as: k & 0x1 << 31
The next bit: k & << 30.
Etc. We call 31 and 30 the "level" of the tree here, so
the level is at most 31. The last level, 0, is used to
select particles rather than branches.

So if the branch is level 31 - 1, it has branches as children.
If it is level 0, it has particles as children.

Level 31 branches are created when the tree is constructed
as m_branches[0,3].
*/

#endif
