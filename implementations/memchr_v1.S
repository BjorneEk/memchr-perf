
#include "asm.h"

EXPORT_FN(memchr_1v)


FN(memchr_1v)
	.cfi_startproc
	dup	v0.16b, w1
LBL(begin):
	cmp	x2, #64
	b.lt	LBL(l1)
	ld1	{v1.16b, v2.16b, v3.16b, v4.16b}, [x0], #64

	cmeq	v5.16b, v0.16b, v1.16b
	cmeq	v6.16b, v0.16b, v2.16b
	cmeq	v7.16b, v0.16b, v3.16b
	cmeq	v8.16b, v0.16b, v4.16b

	//and	v5.16b, v5.16b, v6.16b
	//and	v5.16b, v5.16b, v7.16b
	//and	v5.16b, v5.16b, v8.16b

	umaxv	B0, v5.16b
	umov	w3, v0.b[0]
	cmp	w3, #0xFF
	b.eq	LBL(f64)
	umaxv	B0, v6.16b
	umov	w3, v0.b[0]
	cmp	w3, #0xFF
	b.eq	LBL(f64)
	umaxv	B0, v7.16b
	umov	w3, v0.b[0]
	cmp	w3, #0xFF
	b.eq	LBL(f64)
	umaxv	B0, v8.16b
	umov	w3, v0.b[0]
	cmp	w3, #0xFF
	b.eq	LBL(f64)
	sub	x2, x2, #64
	b	LBL(begin)
LBL(f64):
	sub	x0, x0, #64
LBL(l1):
	cmp	x2, #0
	b.eq	LBL(miss)
	ldrb	w3, [x0], #1
	cmp	w3, w1
	b.eq	LBL(match)
	sub	x2, x2, #1
	b	LBL(l1)
LBL(miss):
	mov	x0, xzr
	b	LBL(end)
LBL(match):
	sub	x0, x0, #1
LBL(end):
	ret
	.cfi_endproc

