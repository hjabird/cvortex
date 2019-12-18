

void bench_first_initialisation(int probsz){
	probsz; /* We don't care about this... */
	cvtx_initialise();
	return;
}

void bench_reinitialisation(int probsz){
	probsz; /* We don't care about this... */
	cvtx_finalise();
	cvtx_initialise();
	return;
}
