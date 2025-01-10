
#include "asm.h"

EXPORT_FN(memchr_4v)


FN(memchr_4v)
	.cfi_startproc
	dup	v0.16b, w1
LBL(loop):
	cmp	x2, #32
	b.lt	LBL(scalar)
	ld1	{v1.16b, v2.16b}, [x0], #32
	cmeq	v1.16b, v1.16b, v0.16b
	cmeq	v2.16b, v2.16b, v0.16b
	umaxp	v3.16b, v1.16b, v1.16b
	umaxp	v4.16b, v2.16b, v2.16b
	fmov	x4, d3
	cbnz	x4, LBL(match32)
	fmov	x4, d4
	cbnz	x4, LBL(match32)
	subs	x2, x2, #32
	b.ge	LBL(loop)
LBL(match32):
	sub	x0, x0, #32
LBL(scalar):
	cbz	x2, LBL(miss)
LBL(scalar_):
	ldrb	w3, [x0], #1
	cmp	w3, w1
	b.eq	LBL(match)
	subs	x2, x2, #1
	b.ge	LBL(scalar_)
LBL(miss):
	mov	x0, xzr
	b	LBL(end)
LBL(match):
	sub	x0, x0, #1
LBL(end):
	ret
	.cfi_endproc

