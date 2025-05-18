FUNCTION SHA1_TO_STRH : string
VAR_INPUT
    SHA1: ARRAY[0..19] OF BYTE;
END_VAR

VAR
    i: INT;
    tmp: BYTE;
    pt: POINTER TO BYTE;
END_VAR
pt := ADR(SHA1_TO_STRH); (* read output adress to pointer *)

FOR i := 0 TO 19 DO; (* write the 10 hex characters backwards *)
	tmp := SHR(SHA1[i],4);
	PT^ := tmp + SEL(tmp <= 9,87,48);
	pt := pt + 1; (* decrement the pointer *)
	tmp := SHA1[i] AND 2#1111;
	PT^ := tmp + SEL(tmp <= 9,87,48);
	pt := pt + 1; (* decrement the pointer *)
END_FOR;
pt^:= 0; (* write the closing byte for the string *)

(* revision history

ks	6. dec. 2009 rev 1.0
	original version

ks	3. dec. 2011 rev 1.1
	incorrect string size
*)
END_FUNCTION
