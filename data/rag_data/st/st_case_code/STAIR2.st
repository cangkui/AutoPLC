FUNCTION_BLOCK STAIR2
VAR_INPUT
    X : REAL;
    D : REAL;
END_VAR

VAR_OUTPUT
    Y : REAL;
END_VAR
IF D > 0.0 THEN
	IF X >= Y + D OR X <= Y - D THEN Y := FLOOR(X/D) * D; END_IF;
ELSE
	Y := X;
END_IF;



(* revision history
hm	28 jan 2007		rev 1.0
	original version

hm	27. dec 2007	rev 1.1
	changed code for better performance

hm	30. jun. 2008	rev 1.2
	added type conversions to avoid warnings under codesys 3.0																																																																																																																																																																																																																																																											

ks	26. oct. 2008	rev 1.3
	improved code

hm	10. mar. 2009	rev 1.4
	real constants updated to new systax using dot

*)
END_FUNCTION_BLOCK
