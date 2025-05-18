FUNCTION_BLOCK DRIVER_4C
VAR_INPUT
    IN : BOOL;
    RST : BOOL;
    Timeout : TIME;
    SX : ARRAY[1..7] OF BYTE := [1, 3, 7, 15];
END_VAR

VAR_OUTPUT
    SN : INT;
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
END_VAR

VAR
    off : TON;
    edge : BOOL;
END_VAR
IF RST OR off.Q THEN
	SN := 0;
ELSIF IN AND NOT edge THEN
	SN := SN + 1;
	IF SN > 7 OR SX[SN] = 0 THEN SN := 0; END_IF;
END_IF;
edge := in;
IF SN > 0 THEN
	Q0 := SX[SN].0;
	Q1 := SX[SN].1;
	Q2 := SX[SN].2;
	Q3 := SX[SN].3;
ELSE
	Q0 := FALSE;
	Q1 := FALSE;
	Q2 := FALSE;
	Q3 := FALSE;
END_IF;

(* maximaum timeout *)
IF timeout > t#0s THEN off(in := SN > 0, PT := Timeout); END_IF;


(* revision history
hm	23. mar. 2009	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
