#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "perf.h"
#include "impl.h"

#define TC(nm) (test_t){.fn=nm, .name=# nm}
static test_t TEST_ARR[] = {(test_t){.fn=memchr, .name="libc"},
	//TC(memchr_1v),
	TC(memchr_2v),
	TC(memchr_3v),
	TC(memchr_4v),
	TC(memchr_5v)};

#define TIME(...)	({							\
	double _time_res;							\
	do {									\
		struct timespec _time_start, _time_end;				\
		long _time_seconds, _time_nanoseconds;				\
		clock_gettime(CLOCK_MONOTONIC, &_time_start);			\
		__VA_ARGS__;							\
		clock_gettime(CLOCK_MONOTONIC, &_time_end);			\
		_time_seconds = _time_end.tv_sec - _time_start.tv_sec;		\
		_time_nanoseconds = _time_end.tv_nsec - _time_start.tv_nsec;	\
		if (_time_nanoseconds < 0) {					\
			_time_seconds--;					\
			_time_nanoseconds += 1000000000;			\
		}								\
		_time_res = _time_seconds + _time_nanoseconds / 1e9;		\
	} while(0);								\
	_time_res;								\
})

size_t	TEST_COUNT = (sizeof TEST_ARR / sizeof(test_t));

void time_all(FILE *out, const void *str, int c, size_t len, int needle, bool verb)
{
	size_t i;
	double t;
	char *s, *se;
	char buff[4096];
	se = memchr(str, c, len);
	for (i = 0; i < TEST_COUNT - 1; ++i) {
		t = TIME({
			s = TEST_ARR[i].fn(str, c, len);
		});
		fprintf(out, "%lf ", t);
		if (verb) {
			if (len < 4096)
				sprintf(buff, "%s(\"%s\", '%c', %lu)", TEST_ARR[i].name, (char*)str, c, len);
			else
				sprintf(buff, "%s(\"...\", '%c', %lu)", TEST_ARR[i].name, c, len);
			if (s != se)
				fprintf(stderr, "%-50s %s '%c' != '%c' & %p != %p\n", buff, s == se ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", s == NULL ? '~' : *s, needle, (void*)s, (void*)se);
		}
	}
	t = TIME({
		s = TEST_ARR[TEST_COUNT - 1].fn(str, c, len);
	});
	fprintf(out, "%lf, ", t);
	if (verb) {
		if (len < 4096)
			sprintf(buff, "%s(\"%s\", '%c', %lu)", TEST_ARR[i].name, (char*)str, c, len);
		else
			sprintf(buff, "%s(\"...\", '%c', %lu)", TEST_ARR[i].name, c, len);
		if (s != se)
			fprintf(stderr, "%-50s %s '%c' != '%c' & %p != %p\n", buff, s == se ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m", s == NULL ? '~' : *s, needle, (void*)s, (void*)se);
	}
}

void put_header(FILE *f, size_t ntests)
{
	fprintf(f, "%lu %lu\n", ntests, TEST_COUNT);
	size_t i;

	for (i = 0; i < TEST_COUNT - 1; i++) {
		fprintf(f, "%s ", TEST_ARR[i].name);
	}
	fprintf(f, "%s\n", TEST_ARR[TEST_COUNT - 1].name);
}

