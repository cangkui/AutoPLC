FUNCTION SDD : REAL
VAR_INPUT
    T : REAL;
    ICE : BOOL;
END_VAR
IF ice THEN
	SDD := 611.153 * EXP(22.4433 * T / (272.186 + T));
ELSE
	SDD := 611.213 * EXP(17.5043 * T / (241.2 + T));
END_IF;



(* revision history

hm 29 oct 2007		rev 1.0
	original version

hm	2. dec 2007		rev 1.1
	changed code for better performance
*)
END_FUNCTION
