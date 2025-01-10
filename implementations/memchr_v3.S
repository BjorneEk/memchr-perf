
#include "asm.h"

EXPORT_FN(memchr_3v)


FN(memchr_3v)
	.cfi_startproc
	dup	v0.16b, w1
LBL(loop):
	cmp	x2, #16
	b.lt	LBL(scalar)
	ldr	q1, [x0], #16
	cmeq	v1.16b, v1.16b, v0.16b
	umaxp	v3.16b, v1.16b, v1.16b
	fmov	x4, d3
	cbnz	x4, LBL(match16)
	subs	x2, x2, #16
	b.ge	LBL(loop)
LBL(match16):
	sub	x0, x0, #16
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

