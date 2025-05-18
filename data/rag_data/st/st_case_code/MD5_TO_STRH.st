FUNCTION MD5_TO_STRH : string
VAR_INPUT
    MD5 : ARRAY[0..15] OF BYTE;
END_VAR

VAR
    i : INT;
    tmp : BYTE;
    pt : POINTER TO BYTE;
END_VAR
pt := ADR(MD5_TO_STRH); (* read output adress to pointer *)

FOR i := 0 TO 15 DO; (* write the 8 hex characters backwards *)
	tmp := SHR(MD5[i],4);
	PT^ := tmp + SEL(tmp <= 9,87,48);
	pt := pt + 1; (* decrement the pointer *)
	tmp := MD5[i] AND 2#1111;
	PT^ := tmp + SEL(tmp <= 9,87,48);
	pt := pt + 1; (* decrement the pointer *)
END_FOR;
pt^:= 0; (* write the closing byte for the string *)

(* revision history

ks	6. dec. 2009 rev 1.0
	original version

*)
END_FUNCTION
