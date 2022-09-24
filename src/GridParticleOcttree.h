#ifndef CVTX_GRIDPARTICLEOCTTREE_H
#define CVTX_GRIDPARTICLEOCTTREE_H
/*============================================================================
GridParticleOcttree.h

A set of vortex particles on a grid represented by a sparse octtree.

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

#include <bsv/bsv_V3f.h>

#include "UIntKey96.h"

class GridParticleOcttree;
class GridParticleOcttreeBranch;

class GridParticleOcttreeBranch { /* Used only by GridParticleOcttree */
public:
	GridParticleOcttreeBranch();
	GridParticleOcttreeBranch(int level, UIntKey96 key);
protected:
	int m_level;
	UIntKey96 m_key;	/* Partial key according to m_level */
	std::array<int64_t, 8> m_child_idxs;
	friend class GridParticleOcttree;
};


class GridParticleOcttree {
public:
	GridParticleOcttree();

	/* Adds vorticity str at location given by key. 
	It is added to any vorticity already at that grid point.*/
	void add_particle(UIntKey96 key, bsv_V3f str);
	void add_particles(
		std::vector<UIntKey96> key, std::vector<bsv_V3f> str);

	/* The number of particles within the tree. */
	size_t number_of_particles();

	/* Get a load of index / strength pairs is a long list. 
	Main method to get particles out of the tree. */
	void flatten_tree(UIntKey96* idxs, bsv_V3f* strs, int max_particles);
	
	/* Merge another GridParticleOctree into this one. */
	void merge_in(const GridParticleOcttree&);

	/* Empty the tree of all particles and branches (except base 
	branches). */
	void clear();

protected:
	/* Branches refer to their children by index, deindexed in this object. */
	std::vector<GridParticleOcttreeBranch> m_branches;
	std::vector<bsv_V3f> m_vorticities;
};


/* 
HOW IS INDEXING IMPLEMENTED HERE?

UIntKey96 has 3 uint32_t subkeys.
The biggest bit of these can be found as: k & 0x1 << 31
The next bit: k & << 30.
Etc. We call 31 and 30 the "level" of the tree here, so
the level is at most 31. The last level, 0, is used to
select particles rather than branches. 

So if the branch is level 31 - 1, it has branches as children.
If it is level 0, it has particles as children.

Level 31 branches are created when the tree is constructed
as m_branches[0,7].
*/

#endif
