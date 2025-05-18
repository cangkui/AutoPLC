FUNCTION BAND_B : BYTE
VAR_INPUT
    X : BYTE;
    B : BYTE;
END_VAR
IF X < B THEN
	BAND_B := 0;
ELSIF X > BYTE#255-B THEN
	BAND_B := 255;
ELSE
	BAND_B := X;
END_IF;



(* revision history
hm	21. nov. 2008	rev 1.0
	original version

*)
END_FUNCTION
