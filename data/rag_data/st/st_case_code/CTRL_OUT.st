FUNCTION_BLOCK CTRL_OUT
VAR_INPUT
    CI : REAL;
    OFFSET : REAL;
    MAN_IN : REAL;
    LIM_L : REAL;
    LIM_H : REAL;
    MANUAL : BOOL;
END_VAR
VAR_OUTPUT
    Y : REAL;
    LIM : BOOL;
END_VAR
Y := SEL(MANUAL, CI, MAN_IN) + OFFSET;

(* Limit the output *)
IF Y > LIM_L AND Y < LIM_H THEN
	LIM := FALSE;
ELSE
	Y := LIMIT(LIM_L, Y, LIM_H);
	LIM := TRUE;
END_IF;



(* revision history
hm 	2. jun. 2008 	rev 1.0
	original version

hm	5. nov. 2008	rev 1.1
	optimized code
*)
END_FUNCTION_BLOCK
