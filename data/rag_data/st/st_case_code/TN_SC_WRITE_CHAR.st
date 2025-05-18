FUNCTION_BLOCK TN_SC_WRITE_CHAR
VAR
    in_Offset : INT;
    Iin_Y : INT;
    Iin_X : INT;
    Iby_CHAR : BYTE;
    Xus_TN_SCREEN : us_TN_SCREEN;
END_VAR
IF TN_SC_XY_ERROR(Iin_X,Iin_Y) THEN RETURN; END_IF;
in_Offset := (Iin_Y * 80) + Iin_X;
Xus_TN_SCREEN.bya_CHAR[in_Offset] := Iby_CHAR; (* Charakter schreiben *)

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
