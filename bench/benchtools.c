/*============================================================================
benchtools.c

Some common tools for the dodgy self contained benchmarking system.

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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "benchtools.h"

char m_test_types[2048], m_test_funcs[2048], m_test_scale[2048];
int m_test_repeats;

float mrandf(float maxf)
{
	return (float)rand() / (float)(RAND_MAX / maxf);
}

void named_bench(char* file_name, int line_no, char* name, void (func)(int), int repr, int probsz) {
	clock_t start, end, diff, sum, sqsum, min, max, ave;
	sum = 0; sqsum = 0; min = 99999; max = 0, ave = 0;
	int i;
	if (!should_run_test(name)) { return; }

	printf("Running test %s for problem size %i (%i repeats)...\n", name, probsz, repr);
	for (i = 0; i < repr; ++i) {
		start = clock();
		func(probsz);
		end = clock();
		diff = end - start;
		sum += diff;
		sqsum += diff * diff;
		min = min < diff ? min : diff;
		max = max > diff ? max : diff;
	}
	printf("\tTest name:\t%s\n", name);
	printf("\tFile:\t\t%s\n", file_name);
	printf("\tLine no.:\t%i\n", line_no);
	printf("\tProb. size:\t%i\n", probsz);
	printf("\tRepeats:\t%i\n", repr);
	printf("\tAverage:\t%i (msec)\n", sum * 1000 / (CLOCKS_PER_SEC * repr));
	printf("\tMinimum:\t%i (msec)\n", min * 1000 / CLOCKS_PER_SEC);
	printf("\tMaximum:\t%i (msec)\n", max * 1000 / CLOCKS_PER_SEC);
	printf("\n");
	return;
}

int parse_command_args(int argc, char* argv[]) {
	char available_types[] = "P2D P3D F3D init";
	char available_funcs[] =
		"vel-gaussian-cpu vel-gaussian-gpu dvort-gaussian-cpu dvort-gaussian-gpu "
		"vel-singular-cpu vel-singular-gpu dvort-singular-cpu dvort-singular-gpu "
		"vel-planetary-cpu vel-planetary-gpu dvort-planetary-cpu dvort-planetary-gpu "
		"vel-winckelmans-cpu vel-winckelmans-gpu dvort-winckelmans-cpu dvort-winckelmans-gpu "
		"viscdvort-winckelmans-cpu viscdvort-winckelmans-gpu "
		"viscdvort-gaussian-cpu viscdvort-gaussian-gpu "
		"vort-gaussian-cpu vort-singular-cpu vort-planetary-cpu vort-winckelmans-cpu "
		"vort-gaussian-gpu vort-singular-gpu vort-planetary-gpu vort-winckelmans-gpu "
		"redistribute-lambda0 redistribute-lambda1 redistribute-lambda2 redistribute-lambda3 "
		"redistribute-m4p cold_initialisation reinitialisation cold reinit";
	char available_scales[] = "vsmall small medium large vlarge huge";

	m_test_repeats = 1;
	strcpy(m_test_types, "");
	strcpy(m_test_funcs, "");
	strcpy(m_test_scale, "");

	int i = 1, tmpi;
	int good = 1;
	char tmpc[2048];
	while (argc > i) {
		/* Set the types that are investigated */
		if (!strcmp(argv[i], "-types")) {
			++i;
			for (; i < argc; ++i) {
				if (argv[i][0] == '-') {
					break;
				}
				else if (token_in_string(argv[i], available_types)) {
					strcat(m_test_types, argv[i]);
					strcat(m_test_types, " ");
				}
				else {
					printf("Unknown type %s. Known types are %s.\n",
						argv[i], available_types);
					good = 0; break;
				}
			}
		}
		/* Set the functions that are run */
		else if (!strcmp(argv[i], "-funcs")) {
			++i;
			for (; i < argc; ++i) {
				if (argv[i][0] == '-') {
					break;
				}
				else if (token_in_string(argv[i], available_funcs)) {
					strcat(m_test_funcs, argv[i]);
					strcat(m_test_funcs, " ");
				}
				else {
					printf("Unknown function %s. Known functions are %s.\n",
						argv[i], available_funcs);
					good = 0; break;
				}
			}
		}
		/* Choose test scale */
		else if (!strcmp(argv[i], "-scales")) {
			++i;
			for (; i < argc; ++i) {
				if (argv[i][0] == '-') {
					break;
				}
				else if (token_in_string(argv[i], available_scales)) {
					strcat(m_test_scale, argv[i]);
					strcat(m_test_scale, " ");
				}
				else {
					printf("Unknown scale %s. Known scales are %s.\n",
						argv[i], available_scales);
					good = 0; break;
				}
			}
		}
		/* Set the number of repeats of each benchmark */
		else if (!strcmp(argv[i], "-repeats")) {
			++i;
			if (i < argc) {
				if (sscanf(argv[i], "%i%1s", &tmpi, tmpc) < 1) {
					good = 0; break;
				}
				m_test_repeats = tmpi;
				if (m_test_repeats < 1) {
					printf("Test repeats must be more than 0.");
					good = 0; break;
				}
				++i;
			}
			else { good = 0; break; }
		}
		else if (!strcmp(argv[i], "-help")) {
			good = 0; break;
		}
		else {
			good = 0;
			break;
		}
	}

	if (good != 1) {
		printf("Bad arguments!\n"
			"Expecting to see:\n"
			"\tall_bench -types [types] -funcs [funcs] -scales [scales] -repeats 10\n\n"
			"Where available types are:\n"
			"%s\n\nAvailable funcs are:\n%s\n\nAvailable scales are:\n%s\n\n",
			available_types, available_funcs, available_scales);
	}
	return good;
}

int should_run_test(char* name) {
	char* token;
	char workspace[2048];	/* strtok modifies reference string. */
	int yes = 1;
	/* Format: "TYPE FUNC VARIATION" */
	strcpy(workspace, name);
	token = strtok(workspace, " ");
	if (strlen(m_test_types) > 0) {		/* Nothing means yes automatically */
		if (!token_in_string(token, m_test_types)) { yes = 0; }
	}
	token = strtok(NULL, " ");
	if (yes && strlen(m_test_funcs) > 0 && token != NULL) {/* Nothing means yes automatically */
		if (!token_in_string(token, m_test_funcs)) { yes = 0; }
	}
	token = strtok(NULL, " ");
	if (yes && strlen(m_test_scale) > 0 && token != NULL) {/* Nothing means yes automatically */
		if (!token_in_string(token, m_test_scale)) { yes = 0; }
	}
	return yes;
}

int token_in_string(char* token, char* ref_str) {
	int i = 0;
	if (strstr(ref_str, token) != NULL) {
		i = 1;
	}
	return i;
}

void print_test_res(char* name, double* times, long long int mincounts) {
	double min, max, mean;
	int i;
	min = max = mean = times[0];
	for (i = 1; i < m_test_repeats; ++i) {
		min = min > times[i] ? times[i] : min;
		max = max < times[i] ? times[i] : max;
		mean += times[i];
	}
	mean /= (double)m_test_repeats;
	printf("%-35s Min: %.3es   Max: %.3es   Mean: %.3es   MinCounts: %lli\n",
		name, min, max, mean, mincounts);
	return;
}

int test_repeats() {
	return m_test_repeats;
}

char* test_types() {
	return m_test_types;
}

char* test_funcs() {
	return m_test_funcs;
}

char* test_scale() {
	return m_test_scale;
}

