FUNCTION_BLOCK DIMM_I
VAR_INPUT
    SET : BOOL;
    VAL : BYTE := 255;
    IN : BOOL;
    RST : BOOL;
    T_DEBOUNCE : TIME := TIME#10ms;
    T_RECONFIG : TIME := TIME#10s0ms;
    T_ON_MAX : TIME := TIME#0ms;
    T_DIMM_START : TIME := TIME#1s0ms;
    T_DIMM : TIME := TIME#3s0ms;
    MIN_ON : BYTE := 50;
    MAX_ON : BYTE := 255;
    SOFT_DIMM : BOOL := TRUE;
    DBL_toggle : BOOL;
    RST_OUT : BOOL;
END_VAR_INPUT

VAR_OUTPUT
    Q : BOOL := FALSE;
    DBL : BOOL;
END_VAR_OUTPUT

VAR_IN_OUT
    OUT : BYTE;
END_VAR_IN_OUT

VAR
    config : SW_RECONFIG;
    decode : CLICK_MODE;
    t3 : OSCAT_BASIC.TON;
    dim : OSCAT_BASIC._RMP_B;
    dir : BOOL;
END_VAR
(* the input is first sent through auto reconfiguration and debouncing *)
config(IN := IN, TD := T_DEBOUNCE, TR := T_RECONFIG);
decode(in := config.Q, T_LONG := T_DIMM_START);

IF RST THEN
	IF rst_out THEN out := 0; END_IF;
	Q := FALSE;
	dir := out > 127;
ELSIF set THEN
	out := val;
	Q := TRUE;
	(* set direction to up when value < 127 otherwise set dir down dir is reversed because next action will reverse again *)
	dir := out > 127;
ELSIF decode.SINGLE THEN
	(* a single click reverses output Q *)
	Q := NOT Q;
	(* when dimmer is turned on we need to limit out to min and max limits *)
	IF Q THEN	out := LIMIT(MAX(MIN_ON,1), out, MAX_ON);	END_IF;
	(* set the appropriate direction of dimmer dir is reversed because next action will reverse again *)
	dir := out > 127;
ELSIF decode.TP_LONG THEN
	IF NOT Q THEN
		IF SOFT_DIMM THEN
			OUT := 1;
			dir := TRUE;
		ELSE
			OUT := LIMIT(MAX(MIN_ON,1), out, MAX_ON);
			DIR := out < 127;
		END_IF;
		Q := TRUE;
	ELSE
		(* reverse direction with every long click *)
		dir := NOT dir;
	END_IF;
END_IF;

(* set the double click output *)
IF NOT dbl_toggle THEN dbl := FALSE; END_IF;
IF decode.DOUBLE THEN dbl := NOT dbl; END_IF;

(* while dimming is active ramp the output out up or down *)
dim(DIR := dir, e := decode.LONG AND Q, TR := T_DIMM, rmp := out);

(* reverse direction when limits are reached *)
IF out = 0 THEN
	dir := TRUE;
ELSIF out = 255 THEN
	dir := FALSE;
END_IF;

(* limit the maximum runtime *)
IF t_ON_MAX > t#0s THEN
	t3(in := q, pt := T_ON_MAX);
	Q := Q XOR t3.Q;
END_IF;




(* revision history

hm 22.1.2007		rev 1.1
	deleted unused structure ramp

hm 2.feb 2007		rev 1.2
	changed structure to read time() at the beginning for better time consistency if a higher priorized tak interrupts dimm_i
	added statements in case structure for state 1 to allow for short 1 cycle input pulses

hm 	15.9.2007		rev 1.3
	replaced Time() with T_PLC_MS for compatibility and performance reasons

hm	30. sep. 2008	rev 1.4
	added Input VAL to supply value for set input
	added setup parameter init_val

hm	22. oct. 2008	rev 2.0
	new code and features

hm	14. nov. 2008	rev 2.1
	added soft_dimm feature

hm	16. nov. 2008	rev 2.2
	added reset out feature

hm	26. jan. 2009	rev 2.3
	dimming will also reverse at out = 0
	rst will override set
*)
END_FUNCTION_BLOCK
