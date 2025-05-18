FUNCTION WORD_TO_RANGE : REAL
VAR_INPUT
    X : WORD;
    low : REAL;
    high : REAL;
END_VAR
WORD_TO_RANGE := (high - low) * WORD_TO_REAL(X) * 0.00001525902189669640 + low;


(* revision history
hm	9. jan 2008	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	improved code
*)
END_FUNCTION
