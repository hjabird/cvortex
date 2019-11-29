/*============================================================================
cstrigify.c

Convert a C-like language source file to a C string. Shrinks input.
Part of the CVortex toolchain.

Usage:
cstringify inputfile.ocl outputfile.ocl [-d -c -w -n]

Optional flags:
-d is debug: The original file, but as a string.
-c is include comments: Includes original comments in strigified output.
-w is include all whitespace: Does not shrink original whitespace at all.
-n is line numbers: Include original line numbers as comments in output.

Copyright(c) 2019 HJA Bird

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

int g_is_debug;				/* default false */
int g_include_comments;		/* default false */
int g_inc_whitespace;		/* default false */

void print_help_message(void);
int set_global_flags(int argc, char* argv[]);
int open_input_file(int argc, char* argv[], FILE** input_file);

int main(int argc, char* argv[])
{
	FILE* input_file = NULL, output_file = NULL;
	size_t position = 0;
	int good = 0;
	good = set_global_flags(argc, argv);
	good = open_input_file(argc, argv, &input_file);
	
	/* Create output buffer. */
	/* Copy input to output buffer, but stringified */
	/* Close input file */
	/* Write output buffer to file */
}

int set_global_flags(int argc, char* argv[])
{
	int good = 1;
	int i;
	g_is_debug = 0;
	g_include_comments = 0;
	g_inc_whitespace = 0;
	
	if(argc < 3){
		printf("Expected more arguments...\n");
		print_help_message();
	}

	for(i = 1; i < argc && good == 1; ++i){
		if(argv[i][0] == '-')
		{/* If it starts with a '-' like -d then its a flag */
			if(argv[i][1] == 'd'){
				g_is_debug = 1;
				g_include_comments = 1;
				g_inc_whitespace = 1;
			} else if(argv[i][1] == 'c'){
				g_include_comments = 1;
			} else if(argv[i][1] == 'w'){
				g_inc_whitespace = 1;
			} else if(argv[i][1] == 'n'){
				g_inc_line_no = 1;
			} else {
				printf("Unknown flag: %s\n", argv[i]);
				print_help_message();
				good = 0;
			}	
		}
	}
	return good;
}

int open_input_file(int argc, char* argv[], FILE** input_file){
	int good = 1;
	if(argc < 2){
		print_help_message();
		good = 0;
		return good;
	}
	*FILE = fopen(argv[1], "r");
	if(*FILE == NULL || errno != 0){
		good = 0;
		printf("Failed to open input file %s\n", argv[1]);
	}
	return good;
}

void print_help_message(void)
{
	str = 	"cstringify - turn c-lang like source to a string\n"
			"HJAB 2019\n"
			"Usage:\n"
			"\tcstringify inputfile outputfile [optional flags]\n"
			"\tWhere possible optional flags are\n"
			"\t\t-d\tdebug: Make the as similar to input as possible.\n"
			"\t\t-c\tinclude comments: Include the input's comments.\n"
			"\t\t-w\tinclude whitespace: Don't shrink the whitespace of the input.\n"
			"\t\t-n\tdebug: Add input line no. comments to output.\n"
			"\n";
	printf("%s", str);
	return;
}

