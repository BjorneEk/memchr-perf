#ifndef _ASM_H_
#define _ASM_H_
.section	__TEXT,__text,regular,pure_instructions
.p2align	8

#define LBL(name) .L ## name

#define EXPORT_FN(name) .globl _ ## name

#define FN(name) _ ## name ## :

#endif /* _ASM_H_ */