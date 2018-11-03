#include "Vec3f.h"
/*============================================================================
Vec3f.c

Basic representation of a vector in 3d space using floats.

Copyright(c) 2018 HJA Bird

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

#include <math.h>

float cvtx_Vec3f_abs(const cvtx_Vec3f self){
	float a, b, c;
	a = powf(self.x[0], 2);
	b = powf(self.x[1], 2);
	c = powf(self.x[2], 2);
	return sqrtf(a + b + c);	
}

float cvtx_Vec3f_dot(const cvtx_Vec3f self, const cvtx_Vec3f other){
	float a, b, c;
	a = self.x[0] * other.x[0];
	b = self.x[1] * other.x[1];
	c = self.x[2] * other.x[2];
	return a + b + c;
}

cvtx_Vec3f cvtx_Vec3f_zero(void){
	cvtx_Vec3f ret;
	ret.x[0] = 0.0;
	ret.x[1] = 0.0;
	ret.x[2] = 0.0;
	return ret;
}

cvtx_Vec3f cvtx_Vec3f_cross(const cvtx_Vec3f self, const cvtx_Vec3f other){
	cvtx_Vec3f ret;
	ret.x[0] = self.x[1] * other.x[2] - 
		self.x[2] * other.x[1];
	ret.x[1] = self.x[2] * other.x[1] - 
		self.x[1] * other.x[2];
	ret.x[2] = self.x[0] * other.x[1] - 
		self.x[1] * other.x[0];
	return ret;	
}

cvtx_Vec3f cvtx_Vec3f_plus(const cvtx_Vec3f self, const cvtx_Vec3f other){
	cvtx_Vec3f ret;
	ret.x[0] = self.x[0] + other.x[0];
	ret.x[1] = self.x[1] + other.x[1];
	ret.x[2] = self.x[2] + other.x[2];
	return ret;
}	

cvtx_Vec3f cvtx_Vec3f_minus(const cvtx_Vec3f self, const cvtx_Vec3f other){
	cvtx_Vec3f ret;
	ret.x[0] = self.x[0] - other.x[0];
	ret.x[1] = self.x[1] - other.x[1];
	ret.x[2] = self.x[2] - other.x[2];
	return ret;
}	

cvtx_Vec3f cvtx_Vec3f_uminus(const cvtx_Vec3f self){
	cvtx_Vec3f ret;
	ret.x[0] = -self.x[0];
	ret.x[1] = -self.x[1];
	ret.x[2] = -self.x[2];
	return ret;
}	

cvtx_Vec3f cvtx_Vec3f_mult(const cvtx_Vec3f self, const float multiplier){
	cvtx_Vec3f ret;
	ret.x[0] = self.x[0] * multiplier;
	ret.x[1] = self.x[1] * multiplier;
	ret.x[2] = self.x[2] * multiplier;
	return ret;
}	

cvtx_Vec3f cvtx_Vec3f_div(const cvtx_Vec3f self, const float div){
	cvtx_Vec3f ret;
	ret.x[0] = self.x[0] / div;
	ret.x[1] = self.x[1] / div;
	ret.x[2] = self.x[2] / div;
	return ret;
}	

bool cvtx_Vec3f_isequal(const cvtx_Vec3f self, const cvtx_Vec3f other){
	bool retv;
	if(	(self.x[0] == other.x[0]) &&
		(self.x[1] == other.x[1]) &&
		(self.x[2] == other.x[2])){

		retv = true;
	} else {
		retv = false;
	}
	return retv;
}

bool cvtx_Vec3f_isnequal(const cvtx_Vec3f self, const cvtx_Vec3f other){
	return !cvtx_Vec3f_isequal(self, other);
}
