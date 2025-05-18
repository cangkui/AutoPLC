FUNCTION RANGE_TO_WORD : WORD
VAR_INPUT
    X : REAL;
    low : REAL;
    high : REAL;
END_VAR
RANGE_TO_WORD := DINT_TO_WORD(TRUNC((LIMIT(low,X,high)-low) * 65535.0 / (high - low)));


(* revision history
hm	9. jan 2008		rev 1.0
	original version

*)
END_FUNCTION
