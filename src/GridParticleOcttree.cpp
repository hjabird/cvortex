#include "GridParticleOcttree.h"
/*============================================================================
GridParticleOcttree.cpp

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

void GridParticleOcttree::add_particle(UIntKey96 key, bsv_V3f str)
{
	if (bsv_V3f_isequal(str, bsv_V3f_zero())) { return; }
	int level = 31; /* Top level of the tree. */
	uint32_t local_idx;
	int nxt_branch_idx, particle_idx; /* Needs to represent -1 too. */
	GridParticleOcttreeBranch *branch;
	size_t this_branch_idx;
	local_idx = key.lidx(level);
	branch = &m_branches[local_idx];
	nxt_branch_idx = local_idx;
	/* Traverse branches */
	for (level = 31; level > 1; --level) {
		assert(branch->m_level == level);
		local_idx = key.lidx(level - 1);
		this_branch_idx = nxt_branch_idx;
		nxt_branch_idx = branch->m_child_idxs[local_idx];
		if (nxt_branch_idx == -1) { /* -1 aka doesn't exist */
			UIntKey96 pkey = UIntKey96::partial_key(local_idx, level - 1);
			m_branches.emplace_back(level - 1, 
				branch->m_key | UIntKey96::partial_key(local_idx, level-1));
			int tmp = m_branches.size() - 1;
			/* Can't depend that point is the same if vector resizes. */
			branch = &m_branches[this_branch_idx];
			branch->m_child_idxs[local_idx] = tmp;
			nxt_branch_idx = tmp;
		}
		branch = &m_branches[nxt_branch_idx];
	}
	assert(branch->m_level == 1);
	local_idx = key.lidx(0);
	particle_idx = branch->m_child_idxs[local_idx];
	if (particle_idx == -1) { /* -1 aka doesn't exist */
		m_vorticities.push_back(str);
		branch->m_child_idxs[local_idx] = m_vorticities.size()-1;
	}
	else {
		m_vorticities[particle_idx] = bsv_V3f_plus(
			m_vorticities[particle_idx], str);
	}
	return;
}

void GridParticleOcttree::add_particles(std::vector<UIntKey96> key, std::vector<bsv_V3f> str)
{
	assert(key.size() == str.size());
	int nk = key.size(), cidx;
	std::array<UIntKey96, 33> k_stack; /* Stack of branch keys. */
	std::array<uint32_t, 33> b_stack; /* Stack of working branch idxs. */
	int level = 32; /* Top level of the tree. */
	uint32_t local_idx;
	/* Traverse branches */
	for (int i = 0; i < nk; ++i) {	
		UIntKey96 tkey = key[i];
		if (bsv_V3f_isequal(str[i], bsv_V3f_zero())) { continue; }
		while (true) {
			if (level == 32) { /* Special case. */
				local_idx = tkey.lidx(31);
				b_stack[level-1] = local_idx;
				k_stack[level-1] = m_branches[local_idx].m_key;
				level = 31;
				continue;
			}
			int match_ext = 32 - tkey.matching_leading_bits(k_stack[level]);
			if (level == 1 && match_ext <= level) { /* Insert particle. */
				local_idx = tkey.lidx(0);
				cidx = m_branches[b_stack[level]].m_child_idxs[local_idx];
				if (cidx == -1) {
					m_vorticities.push_back(str[i]);
					m_branches[b_stack[level]].m_child_idxs[local_idx] = 
						m_vorticities.size() - 1;
				} else {
					m_vorticities[cidx] = bsv_V3f_plus(
						m_vorticities[cidx], str[i]);
				}
				break; /* DONE! To the next particle. */
			}
			if (match_ext <= level) { /* Find next branch. */
				local_idx = tkey.lidx(level - 1);
				cidx = m_branches[b_stack[level]].m_child_idxs[local_idx];
				if (cidx == -1) {
					UIntKey96 pkey = UIntKey96::partial_key(local_idx, level - 1);
					m_branches.emplace_back(level - 1,
						k_stack[level] | UIntKey96::partial_key(local_idx, level - 1));
					cidx = m_branches.size() - 1;
					m_branches[b_stack[level]].m_child_idxs[local_idx] = cidx;
				}
				b_stack[level-1] = cidx;
				k_stack[level-1] = m_branches[cidx].m_key;
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

size_t GridParticleOcttree::number_of_particles()
{
	return m_vorticities.size();
}

void GridParticleOcttree::flatten_tree(UIntKey96* idxs, bsv_V3f* strs, int max_particles)
{	/* Depth first traverse of tree extracting particles. */
	size_t i_out = 0;	/* Output array index. */
	std::stack<int64_t> bidxs; /* Need to do -1 for first 7 branches. */
	std::stack<GridParticleOcttreeBranch*> branches;
	for (uint32_t i = 0; i < 7; ++i) {	/* Add base branches. */
		bidxs.push(-1);	branches.push(&m_branches[i]);
	}
	bidxs.push(0);	branches.push(&m_branches[7]);
	while (bidxs.size() > 0) {
		assert(branches.size() == bidxs.size());
		uint32_t lvl = branches.top()->m_level;
		if (bidxs.top() == 8) { 
			bidxs.pop(); branches.pop(); 
			if (bidxs.size() == 0) { break; }
			bidxs.top()++;
			continue;
		}
		if (lvl > 1) { /* Traverse branches*/
			int cidx = branches.top()->m_child_idxs[bidxs.top()];
			if (cidx != -1) {
				branches.push(&m_branches[cidx]);
				bidxs.push(0);
			}
			else {
				bidxs.top()++;
			}
		}
		else /* Copy child particles to output. */
		{
			for (uint32_t j = 0; j < 8; j++) {
				int cidx = branches.top()->m_child_idxs[j];
				if (cidx != -1 && i_out < max_particles) {
					idxs[i_out] = branches.top()->m_key |
							UIntKey96::partial_key(j, 0);
					strs[i_out] = m_vorticities[cidx];
					i_out++;
				}
			}
			bidxs.top() = 8;
		}
		if (i_out >= max_particles) { break; }
	}
	return;
}

GridParticleOcttree::GridParticleOcttree()
	: m_vorticities(), m_branches()
{
	m_branches.resize(8);
	for (int i = 0; i < 8; ++i) {
		m_branches[i].m_level = 31;
	}
}

GridParticleOcttreeBranch::GridParticleOcttreeBranch()
	: m_level(-1), m_child_idxs()
{
	for (int i = 0; i < 8; ++i) { m_child_idxs[i] = -1; }
}

GridParticleOcttreeBranch::GridParticleOcttreeBranch(int level, UIntKey96 key)
	: m_level(level), m_child_idxs(), m_key(key)
{
	for (int i = 0; i < 8; ++i) { m_child_idxs[i] = -1; }
	assert(level > 0);
	assert(level < 32);
	assert(m_key.k.x << (32 - m_level) == 0x0);
	assert(m_key.k.y << (32 - m_level) == 0x0);
	assert(m_key.k.z << (32 - m_level) == 0x0);
}


void GridParticleOcttree::merge_in(const GridParticleOcttree &other)
{	/* depth first traverse of other tree, replicating it parts not
	matched bit this tree. */
	std::stack<size_t> bidx1, bidx2; /* this and other's m_branches index. */
	std::stack<int64_t> jidx;
	int insertion_count = 0, merge_count = 0, branch_ins_cnt = 0;
	for (uint32_t i = 0; i < 7; ++i) {	/* Add base branches. */
		bidx1.push(i);	bidx2.push(i); jidx.push(-1);
	}
	bidx1.push(7);	bidx2.push(7);	jidx.push(0);
	while (bidx1.size() > 0) {
		assert((jidx.size() == bidx1.size()) && (jidx.size() == bidx2.size()));
		GridParticleOcttreeBranch *b1 = &m_branches[bidx1.top()];
		GridParticleOcttreeBranch const *b2 = &(other.m_branches[bidx2.top()]);
		assert(b1->m_key == b2->m_key);
		uint32_t lvl = b1->m_level;
		int64_t j = jidx.top();
		if (jidx.top() == 8) {
			bidx1.pop(); bidx2.pop(); jidx.pop();
			if (bidx1.size() == 0) { break; }
			jidx.top()++;
			continue;
		}
		int cidx1 = b1->m_child_idxs[j];
		int cidx2 = b2->m_child_idxs[j];
		if (lvl > 1) { /* Traverse branches*/
			if (cidx2 != -1) {
				if (cidx1 == -1) {
					GridParticleOcttreeBranch const *b2c = 
						&(other.m_branches[b2->m_child_idxs[j]]);
					m_branches.push_back(*b2c);
					int64_t nidx = m_branches.size() - 1;
					b1 = &m_branches[bidx1.top()]; /* Realloc on push_back */
					b1->m_child_idxs[j] = nidx;
					std::fill(m_branches[nidx].m_child_idxs.begin(), 
						m_branches[nidx].m_child_idxs.end(), -1);
					cidx1 = nidx;
					branch_ins_cnt++;
				}
				bidx1.push(cidx1);	bidx2.push(cidx2);	jidx.push(0);
			}
			else {
				jidx.top()++;
			}
		}
		else /* Copy child particles to other. */
		{
			for (uint32_t j = 0; j < 8; j++) {
				cidx1 = b1->m_child_idxs[j];
				cidx2 = b2->m_child_idxs[j];
				if (cidx2 != -1) {
					if (cidx1 == -1) {
						m_vorticities.push_back(
							other.m_vorticities[b2->m_child_idxs[j]]);
						int64_t nidx = m_vorticities.size() - 1;
						b1->m_child_idxs[j] = nidx;
						insertion_count++;
					} else {
						m_vorticities[cidx1] = bsv_V3f_plus(
							m_vorticities[cidx1], other.m_vorticities[cidx2]);
						merge_count++;
					}
				}
			}
			jidx.top() = 8;
		}
	}
	return;
}

void GridParticleOcttree::clear()
{
	m_branches.resize(8);
	m_vorticities.resize(0);
	for (uint32_t i = 0; i < 8; ++i) {
		m_branches[i] = GridParticleOcttreeBranch(
			31, UIntKey96(i & 0x1, i<<1 & 0x1, i<<2 & 0x1)<<32);
	}
	return;
}
