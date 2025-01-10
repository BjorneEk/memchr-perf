
#include "asm.h"

EXPORT_FN(memchr_5v)


FN(memchr_5v)
	.cfi_startproc
	dup	v0.16b, w1
LBL(loop):
	cmp	x2, #64
	b.lt	LBL(scalar)
	ld1	{v1.16b, v2.16b, v3.16b, v4.16b}, [x0], #64
	cmeq	v1.16b, v1.16b, v0.16b
	cmeq	v2.16b, v2.16b, v0.16b
	cmeq	v3.16b, v3.16b, v0.16b
	cmeq	v4.16b, v4.16b, v0.16b
	umaxp	v5.16b, v1.16b, v1.16b
	umaxp	v6.16b, v2.16b, v2.16b
	fmov	x4, d5
	cbnz	x4, LBL(match64)
	umaxp	v7.16b, v3.16b, v3.16b
	fmov	x4, d6
	cbnz	x4, LBL(match64)
	umaxp	v8.16b, v4.16b, v4.16b
	fmov	x4, d7
	cbnz	x4, LBL(match64)
	fmov	x4, d8
	cbnz	x4, LBL(match64)
	subs	x2, x2, #64
	b.ge	LBL(loop)
LBL(match64):
	sub	x0, x0, #64
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

