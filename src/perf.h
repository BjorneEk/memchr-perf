
#ifndef _PERF_H_
#define _PERF_H_

#include <stdlib.h>
#include <stdio.h>

typedef struct test {
	void *(*fn)(const void*, int, size_t);
	const char *name;
} test_t;

extern size_t TEST_COUNT;

void time_all(FILE *out, const void *str, int c, size_t len, int needle, bool verb);

void put_header(FILE *f, size_t ntests);
#endif /* _PERF_H_ */