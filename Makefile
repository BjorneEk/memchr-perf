BIN:=bin
BUILD:=build
SRC:=src
IMPL:=implementations
TARGET:=$(BIN)/target
DATA:=data
PLOTS:=plots
CFLAGS:= -std=c2x -pedantic -Wall -Werror -Wno-newline-eof -Wno-gnu-binary-literal -Wno-gnu-statement-expression-from-macro-expansion -g -O0
ASMFLAGS:= -g

C_SOURCES:=$(wildcard $(SRC)/*.c)
ASM_SOURCES:=$(wildcard $(IMPL)/*.S)

OBJ:=$(patsubst $(IMPL)/%.S, $(BUILD)/%.o, $(ASM_SOURCES))
C_OBJ:=$(patsubst $(SRC)/%.c, $(BUILD)/%.o, $(C_SOURCES))

PLOT_INPUTS:=$(wildcard $(DATA)/*.out)
PLOT_OUTPUT:=$(patsubst $(DATA)/%.out, $(PLOTS)/%.png, $(PLOT_INPUTS))
PLOT_SCRIPT:=plot.py
PY:=python

UNAME:=$(shell uname -s)

ifeq ($(UNAME), Linux)
	DEBUG:=gdb -tui
else
	DEBUG:=lldb
endif

all: $(TARGET) Makefile

.PHONY: clean plot run

OBJ_MSG_PRINTED:=1
C_OBJ_MSG_PRINTED:=1
TARGET_MSG_PRINTED:=1

$(BUILD)/%.o: $(IMPL)/%.S
	@mkdir -p $(dir $@)

	$(if $(filter 0,$(MAKELEVEL)), $(if $(filter 0,$(OBJ_MSG_PRINTED)),, \
	$(eval OBJ_MSG_PRINTED:=0) \
	@echo "\nAssembling implementations"))

	@$(CC) $(ASMFLAGS) -c $< -I $(dir $<) -o $@
	@printf " - %-25s <- %s\n" "$@" " $<"

$(BUILD)/%.o: $(SRC)/%.c
	@mkdir -p $(dir $@)

	$(if $(filter 0,$(MAKELEVEL)), $(if $(filter 0,$(C_OBJ_MSG_PRINTED)),, \
	$(eval C_OBJ_MSG_PRINTED:=0) \
	@echo "\nCompiling object files"))

	@$(CC) $(CFLAGS) -c $< -I $(dir $<) -o $@
	@printf " - %-25s <- %s\n" "$@" " $<"

$(TARGET): $(C_OBJ) $(OBJ)
	@mkdir -p $(dir $@)

	$(if $(filter 0,$(MAKELEVEL)), $(if $(filter 0,$(TARGET_MSG_PRINTED)),, \
	$(eval TARGET_MSG_PRINTED:=0) \
	@echo "\nCompiling target"))

	@$(CC) $(CFLAGS) -o $@ $^
	@printf " - %-25s <- %s\n" "$@" "$^"

run: $(TARGET)
	./$<

debug: $(TARGET)
	$(DEBUG) ./$<

$(PLOTS)/%.png: $(DATA)/%.out
	$(PY) $(PLOT_SCRIPT) $< $@

plot: $(PLOT_OUTPUT)


clean:
	rm -rf $(BIN)
	rm -rf $(BUILD)