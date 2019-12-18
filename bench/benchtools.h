#ifndef CVTX_BENCHTOOLS_H
#define CVTX_BENCHTOOLS_H
/*============================================================================
benchtools.h

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

/* Test running function. */
#define BENCH(N, funcptr, R, S) named_bench(__FILE__, __LINE__, N, funcptr, R, S)
void named_bench(char* file_name, int line_no, char* name, void (func)(int), int repr, int probsz);

/* Test control */
int parse_command_args(int argc, char* argv[]);
int should_run_test(char* name);
int token_in_string(char* token, char* ref_str);

/* Print the test results. Mincounts give indication of timing resolution. */
void print_test_res(char* name, double* times, long long int mincounts);

/* Generate a rand between -maxf and maxf*/
float mrandf(float maxf);

/* Get info on tests to run. */
int test_repeats();
char* test_types();
char* test_funcs();
char* test_scale();

#endif /* CVTX_BENCHTOOLS_H */
