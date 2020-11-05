#include "GridParticleQuadtree.h"
/*============================================================================
GridParticleQuadtree.cpp

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

#include <cassert>
#include <cstdint>
#include <stack>
#include <tuple>

void GridParticleQuadtree::add_particle(UIntKey64 key, float str)
{
	if (str == 0.f) { return; }
	int level = 31; /* Top level of the tree. */
	uint32_t local_idx;
	int nxt_branch_idx; /* Needs to represent -1 too. */
	GridParticleQuadtreeBranch* branch;
	size_t this_branch_idx;
	local_idx = key.lidx(level);
	branch = &m_branches[local_idx];
	nxt_branch_idx = local_idx;
	/* Traverse branches */
	for (level = 31; level > 1; --level) {
		assert(branch->m_level == level);
		local_idx = key.lidx(level - 1);
		this_branch_idx = nxt_branch_idx;
		nxt_branch_idx = branch->m_child_idxs[local_idx].v.u;
		if (branch->m_child_idxs[local_idx].type == 0) { /* aka doesn't exist */
			UIntKey64 pkey = UIntKey64::partial_key(local_idx, level - 1);
			m_branches.emplace_back(level - 1,
				branch->m_key | UIntKey64::partial_key(local_idx, level - 1));
			int tmp = m_branches.size() - 1;
			/* Can't depend that point is the same if vector resizes. */
			branch = &m_branches[this_branch_idx];
			branch->m_child_idxs[local_idx].type = 1; /* Is branch. */
			branch->m_child_idxs[local_idx].v.u = tmp;
			nxt_branch_idx = tmp;
		}
		branch = &m_branches[nxt_branch_idx];
	}
	assert(branch->m_level == 1);
	local_idx = key.lidx(0);
	if (branch->m_child_idxs[local_idx].type == 0) { /* aka doesn't exist */
		branch->m_child_idxs[local_idx].type = 2;
		branch->m_child_idxs[local_idx].v.f = str;
		m_vortex_count += 1;
	}
	else {
		assert(branch->m_child_idxs[local_idx].type == 2); /* aka is particle */
		branch->m_child_idxs[local_idx].v.f += str;
	}
	return;
}

void GridParticleQuadtree::add_particles(std::vector<UIntKey64> key, std::vector<float> str)
{
	assert(key.size() == str.size());
	int nk = key.size(), cidx;
	std::array<UIntKey64, 33> k_stack; /* Stack of branch keys. */
	std::array<uint32_t, 33> b_stack; /* Stack of working branch idxs. */
	int level = 32; /* Top level of the tree. */
	uint32_t local_idx;
	/* Traverse branches */
	for (int i = 0; i < nk; ++i) {
		UIntKey64 tkey = key[i];
		if (str[i] == 0.f) { continue; }
		while (true) {
			if (level == 32) { /* Special case. */
				local_idx = tkey.lidx(31);
				b_stack[level - 1] = local_idx;
				k_stack[level - 1] = m_branches[local_idx].m_key;
				level = 31;
				continue;
			}
			int match_ext = 32 - tkey.matching_leading_bits(k_stack[level]);
			if (level == 1 && match_ext <= level) { /* Insert particle. */
				local_idx = tkey.lidx(0);
				cvalue_t& subbranch = 
					m_branches[b_stack[level]].m_child_idxs[local_idx];
				if (subbranch.type == 0) {
					subbranch.type = 2;
					subbranch.v.f = str[i];
					m_vortex_count += 1;
				}
				else {
					assert(subbranch.type == 2);
					subbranch.v.f += str[i];
				}
				break; /* DONE! To the next particle. */
			}
			if (match_ext <= level) { /* Find next branch. */
				local_idx = tkey.lidx(level - 1);
				cvalue_t *subbranch =
					&(m_branches[b_stack[level]].m_child_idxs[local_idx]);
				if (subbranch->type == 0) { /* AKA no branch*/
					UIntKey64 pkey = UIntKey64::partial_key(local_idx, level - 1);
					m_branches.emplace_back(level - 1,
						k_stack[level] | UIntKey64::partial_key(local_idx, level - 1));
					/* m_branches may be realloced, invalidating pointer. */
					subbranch =
						&(m_branches[b_stack[level]].m_child_idxs[local_idx]);
					subbranch->type = 1;
					subbranch->v.u = m_branches.size() - 1;
				}
				b_stack[level - 1] = subbranch->v.u;
				k_stack[level - 1] = m_branches[subbranch->v.u].m_key;
				level--;
				continue;
			}
			else { /* Jump back up tree. */
				level = match_ext;
			}
		}
	}

	return;
}

size_t GridParticleQuadtree::number_of_particles()
{
	return m_vortex_count;
}

void GridParticleQuadtree::flatten_tree(UIntKey64* idxs, float* strs, int max_particles)
{	/* Depth first traverse of tree extracting particles. */
	size_t i_out = 0;	/* Output array index. */
	std::stack<int64_t> bidxs; /* Need to do -1 for first 3 branches. */
	std::stack<GridParticleQuadtreeBranch*> branches;
	for (uint32_t i = 0; i < 3; ++i) {	/* Add base branches. */
		bidxs.push(-1);	branches.push(&m_branches[i]);
	}
	bidxs.push(0);	branches.push(&m_branches[3]);
	while (bidxs.size() > 0) {
		assert(branches.size() == bidxs.size());
		uint32_t lvl = branches.top()->m_level;
		if (bidxs.top() == 4) {
			bidxs.pop(); branches.pop();
			if (bidxs.size() == 0) { break; }
			bidxs.top()++;
			continue;
		}
		if (lvl > 1) { /* Traverse branches*/
			cvalue_t child = branches.top()->m_child_idxs[bidxs.top()];
			if (child.type == 1) { /* aka is a branch*/
				assert(child.v.u < m_branches.size());
				branches.push(&m_branches[child.v.u]);
				bidxs.push(0);
			}
			else {
				assert(child.type == 0); /* aka nothing there. */
				bidxs.top()++;
			}
		}
		else /* Copy child particles to output. */
		{
			for (uint32_t j = 0; j < 4; j++) {
				cvalue_t child = branches.top()->m_child_idxs[j];
				if (child.type != 0 && i_out < max_particles) {
					assert(child.type == 2); /* Is a particle. */
					idxs[i_out] = branches.top()->m_key |
						UIntKey64::partial_key(j, 0);
					strs[i_out] = child.v.f;
					i_out++;
				}
			}
			bidxs.top() = 4;
		}
		if (i_out >= max_particles) { break; }
	}
	return;
}

GridParticleQuadtree::GridParticleQuadtree()
	: m_branches(), m_vortex_count(0)
{
	m_branches.resize(4);
	for (int i = 0; i < 4; ++i) {
		m_branches[i].m_level = 31;
	}
}

GridParticleQuadtreeBranch::GridParticleQuadtreeBranch()
	: m_level(-1), m_child_idxs()
{
	for (int i = 0; i < 4; ++i) { 
		m_child_idxs[i].type = 0; 
	}
}

GridParticleQuadtreeBranch::GridParticleQuadtreeBranch(int level, UIntKey64 key)
	: m_level(level), m_child_idxs(), m_key(key)
{
	for (int i = 0; i < 4; ++i) { 
		m_child_idxs[i].type = 0; 
	}
	assert(level > 0);
	assert(level < 32);
	assert(m_key.k.x << (32 - m_level) == 0x0);
	assert(m_key.k.y << (32 - m_level) == 0x0);
}


void GridParticleQuadtree::merge_in(const GridParticleQuadtree& other)
{	/* depth first traverse of other tree, replicating it parts not
	matched bit this tree. */
	std::stack<size_t> bidx1, bidx2; /* this and other's m_branches index. */
	std::stack<int64_t> jidx;
	int insertion_count = 0, merge_count = 0, branch_ins_cnt = 0;
	for (uint32_t i = 0; i < 3; ++i) {	/* Add base branches. */
		bidx1.push(i);	bidx2.push(i); jidx.push(-1);
	}
	bidx1.push(3);	bidx2.push(3);	jidx.push(0);
	while (bidx1.size() > 0) {
		assert((jidx.size() == bidx1.size()) && (jidx.size() == bidx2.size()));
		GridParticleQuadtreeBranch* b1 = &m_branches[bidx1.top()];
		GridParticleQuadtreeBranch const* b2 = &(other.m_branches[bidx2.top()]);
		assert(b1->m_key == b2->m_key);
		uint32_t lvl = b1->m_level;
		int64_t j = jidx.top();
		if (jidx.top() == 4) {
			bidx1.pop(); bidx2.pop(); jidx.pop();
			if (bidx1.size() == 0) { break; }
			jidx.top()++;
			continue;
		}
		const cvalue_t& cidx2 = b2->m_child_idxs[j];
		if (lvl > 1) { /* Traverse branches*/
			if (cidx2.type == 1) { /* is a branch */
				if (b1->m_child_idxs[j].type != 1) {
					assert(b1->m_child_idxs[j].type == 0);
					GridParticleQuadtreeBranch const* b2c =
						&(other.m_branches[cidx2.v.u]);
					m_branches.push_back(*b2c);
					int64_t nidx = m_branches.size() - 1;
					b1 = &m_branches[bidx1.top()]; /* Realloc on push_back */
					cvalue_t empty_cvalue;
					empty_cvalue.type = 0;
					std::fill(m_branches[nidx].m_child_idxs.begin(),
						m_branches[nidx].m_child_idxs.end(), empty_cvalue);
					cvalue_t& cidx1 = b1->m_child_idxs[j];
					b1->m_child_idxs[j].type = 1;
					b1->m_child_idxs[j].v.u = nidx;
					branch_ins_cnt++;
				}
				bidx1.push(b1->m_child_idxs[j].v.u);	
				bidx2.push(cidx2.v.u);	jidx.push(0);
			}
			else {
				jidx.top()++;
			}
		}
		else /* Copy child particles to other. */
		{
			for (uint32_t j = 0; j < 4; j++) {
				cvalue_t &cidx1 = b1->m_child_idxs[j];
				const cvalue_t &cidx2 = b2->m_child_idxs[j];
				if (cidx2.type == 2) {
					if (cidx1.type == 0) {
						cidx1.type = 2;
						cidx1.v.f = cidx2.v.f;
						insertion_count++;
					}
					else {
						assert(cidx1.type == 2); /* aka is a particle. */
						cidx1.v.f += cidx2.v.f;
						merge_count++;
					}
				}
			}
			jidx.top() = 4;
		}
	}
	m_vortex_count += insertion_count;
	return;
}

void GridParticleQuadtree::clear()
{
	m_branches.resize(4);
	for (uint32_t i = 0; i < 4; ++i) {
		m_branches[i] = GridParticleQuadtreeBranch(
			31, UIntKey64(i & 0x1, i << 1 & 0x1) << 32);
	}
	return;
}
