FUNCTION RDM2 : INT
VAR_INPUT
    last : INT;
    low : INT;
    high : INT;
END_VAR
RDM2 := DINT_TO_INT(TRUNC(RDM(FRACT(last * math.PI)) * (high - low + 1)) + low);

(* revision history
hm		29. feb 2008		rev 1.0
	original version

hm		18. oct. 2008		rev 1.1
	using math constants

*)
END_FUNCTION
