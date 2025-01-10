
#include "asm.h"

EXPORT_FN(memchr_2v)


FN(memchr_2v)
	.cfi_startproc
	dup	v1.16b, w1
LBL(loop):
	cmp	x2, #16
	b.lt	LBL(scalar)
	ldr	q0, [x0], #16
	cmeq	v0.16b, v0.16b, v1.16b
	umaxp	v3.16b, v0.16b, v0.16b
	fmov	x4, d3
	cbnz	x4, LBL(match16)
	subs	x2, x2, #16
	b.ne	LBL(loop)
LBL(match16):
	sub	x0, x0, #16

LBL(scalar):
	cmp	x2, #0
	b.eq	LBL(miss)
	ldrb	w3, [x0], #1
	cmp	w3, w1
	b.eq	LBL(match)
	sub	x2, x2, #1
	b	LBL(scalar)
LBL(miss):
	mov	x0, xzr
	b	LBL(end)
LBL(match):
	sub	x0, x0, #1
LBL(end):
	ret
	.cfi_endproc

