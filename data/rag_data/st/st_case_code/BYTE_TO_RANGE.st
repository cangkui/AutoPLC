FUNCTION BYTE_TO_RANGE : REAL
VAR_INPUT
    X : BYTE;
    low : REAL;
    high : REAL;
END_VAR
BYTE_TO_RANGE := (high - low) * X / 255.0 + low;


(* revision history
hm	9. jan 2008		rev 1.0
	original version

*)
END_FUNCTION
