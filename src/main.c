#include <stdbool.h>
#include <stdio.h>
#include "perf.h"


#define PRINTABLE_MIN (32)
#define PRINTABLE_MAX (126)

static int rprint(char needle)
{
	int res;

	res = (rand() % ((PRINTABLE_MAX - PRINTABLE_MIN) + 1)) + PRINTABLE_MIN;
	if (res == needle)
		return rprint(needle);
	return res;
}

static int rto(int max)
{
	return rand() % (max + 1);
}

static void rstr(char *s, size_t len, char needle, int density)
{
	size_t i;
	for (i = 0; i < len; i++)
		s[i] = rprint(needle);
	for (i = 0; i < len; i++)
		if (rto(density) == 0)
			s[i] = needle;
	s[len] = '\0';
}

void test_random(size_t stride, size_t cnt, int density, int needle, const char *outname)
{
	char *in;
	size_t i;
	FILE *out;

	out = fopen(outname, "w+");
	in = malloc(stride * cnt + 1);
	put_header(out, cnt);

	for (i = 0; i < cnt; i++) {
		rstr(in, needle, density, (i + 1) * stride);
		time_all(out, in, needle, (i + 1) * stride, needle, true);
	}
	fclose(out);
}

int main(void)
{
	test_random(1000, 100000, 5000, 'X', "data/test_1000_100000_5000.out");
	test_random(1000, 100000, 50, 'X', "data/test_1000_100000_50.out");
}


