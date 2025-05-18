FUNCTION_BLOCK DIMM_2
VAR_INPUT
    SET : BOOL;
    VAL : BYTE := 255;
    I1 : BOOL;
    I2 : BOOL;
    RST : BOOL;
    T_DEBOUNCE : TIME := TIME#10ms;
    T_ON_MAX : TIME := TIME#0ms;
    T_DIMM_START : TIME := TIME#1s0ms;
    T_DIMM : TIME := TIME#3s0ms;
    MIN_ON : BYTE := 50;
    MAX_ON : BYTE := 255;
    RST_OUT : BOOL := TRUE;
    SOFT_DIMM : BOOL := TRUE;
    DBL1_TOG : BOOL;
    DBL2_TOG : BOOL;
    DBL1_SET : BOOL;
    DBL2_SET : BOOL;
    DBL1_POS : BYTE;
    DBL2_POS : BYTE;
END_VAR

VAR_OUTPUT
    Q : BOOL := FALSE;
    D1 : BOOL;
    D2 : BOOL;
END_VAR_OUTPUT

VAR_IN_OUT
    OUT : BYTE;
END_VAR_IN_OUT

VAR
    t1 : OSCAT_BASIC.TOF;
    t2 : OSCAT_BASIC.TOF;
    dc1 : CLICK_MODE;
    dc2 : CLICK_MODE;
    t3 : OSCAT_BASIC.TON;
    dim : OSCAT_BASIC._RMP_B;
END_VAR
(* debouncing for inputs i1 and i2 *)
t1(IN := I1, PT := T_DEBOUNCE);
t2(IN := I2, PT := T_DEBOUNCE);
dc1(in := t1.Q, T_LONG := T_DIMM_START);
dc2(in := t2.Q, T_LONG := T_DIMM_START);


IF RST THEN
	IF RST_OUT THEN out := 0; END_IF;
	Q := FALSE;
	D1 := FALSE;
	D2 := FALSE;
ELSIF set THEN
	OUT := VAL;
	Q := TRUE;
ELSIF dc1.SINGLE THEN
	(* single click on i1 turns the dimmer on *)
	out := LIMIT(MAX(MIN_ON,1), out, MAX_ON);
	Q := TRUE;
ELSIF dc2.SINGLE THEN
	(* single click on i2 turns the dimmer off *)
	Q := FALSE;
ELSIF dc1.TP_LONG THEN
	(* continuous on at i1 dimms up *)
	IF NOT Q THEN
		out := SEL(SOFT_DIMM, LIMIT(MAX(MIN_ON,1), out, MAX_ON), 1);
	END_IF;
	Q := TRUE;
	dim.DIR := TRUE;
ELSIF dc2.TP_LONG THEN
	(* continuous on at i2 dimms down *)
	dim.DIR := FALSE;
END_IF;

(* set the double click output *)
IF NOT DBL1_TOG THEN D1 := FALSE; END_IF;
IF dc1.DOUBLE THEN
	IF DBL1_SET THEN
		OUT := DBL1_POS;
		Q := TRUE;
	ELSE
		D1 := NOT D1;
	END_IF;
END_IF;
IF NOT DBL2_TOG THEN D2 := FALSE; END_IF;
IF dc2.DOUBLE THEN
	IF DBL2_SET THEN
		OUT := DBL2_POS;
		Q := TRUE;
	ELSE
		D2 := NOT D2;
	END_IF;
END_IF;


(* while dimming is active ramp the output out up or down *)
dim(dir := dc2.LONG, e := dc1.LONG OR dc2.LONG , TR := T_DIMM, rmp := out);

(* turn off output if a value of 0 is reached *)
IF out = 0 THEN Q := FALSE; END_IF;

(* limit the maximum runtime *)
IF t_ON_MAX > t#0s THEN
	t3(in := q, pt := T_ON_MAX);
	Q := Q XOR t3.Q;
END_IF;




(* revision history

hm 22. mar. 2009	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
