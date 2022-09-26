/*============================================================================
testmain.c

A dodgy self contained test system for cvortex.

Copyright(c) 2018-2019 HJA Bird

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define TEST(X) test(__FILE__, __LINE__, X)
#define NAMED_TEST(X, Y) named_test(__FILE__, __LINE__, X, Y)

#define SECTION(X) section(X)

int total_tests_passed = 0;
int total_tests_completed = 0;
int section_tests_passed = 0;
int section_tests_completed = 0;
char working_section_name[512] = "";

void test(char* file_name, int line_no, int passed) {
	if (!passed) {
		printf("Test failed:\n\t%s\n\tLine %i\n", file_name, line_no);
		assert(0);
	}
	else {
		section_tests_passed += 1;
	}
	section_tests_completed += 1;
	return;
}

void named_test(char* file_name, int line_no, int passed, char* test_name) {
	if (!passed) {
		printf("Test failed:\n\t%s\n\t%s\n\tLine %i\n", test_name, 
			file_name, line_no);
		assert(0);
	}
	else {
		section_tests_passed += 1;
	}
	section_tests_completed += 1;
	return;
}

void section(char section_name[]) {
	if (section_tests_passed > 0) {
		printf("Passed %i of %i tests in section %s.\n",
			section_tests_passed, section_tests_completed, working_section_name);
	}
	total_tests_passed += section_tests_passed;
	total_tests_completed += section_tests_completed;
	section_tests_passed = 0;
	section_tests_completed = 0;
	strcpy(working_section_name, section_name);
	return;
}

int print_summary() {
	printf("\nComplete!\nPassed %i of %i tests (%i failed).\n",
		total_tests_passed, total_tests_completed,
		total_tests_completed - total_tests_passed);
	return total_tests_completed - total_tests_passed;
}

int mrand(void)
{	/* We want randomish, but reproducable tests. */
	static long mrandv = 0xF0F0F0F0;
	return (((mrandv = mrandv * 214013L + 2531011L) >> 16) &
		0x7fff)%RAND_MAX;
}

#include "testaccelerators.h"
#include "testparticle.h"
#include "testsamecpugpuresultsingle.h"
#include "testsamecpugpuresultmany.h"

int main(int argc, char* argv[]){
	cvtx_initialise();
	testAccelerators();
    testParticle();
	testSameCpuGpuResSingle();
	testSameCpuGpuResMany();
	cvtx_finalise();
	SECTION("");
	return print_summary();
}
