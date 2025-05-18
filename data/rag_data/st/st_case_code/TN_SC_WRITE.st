FUNCTION_BLOCK TN_SC_WRITE
VAR
    bo_Color : BOOL;                // Variable of type BOOL
    in_Offset : INT;                // Variable of type INT
    in_Char_Index : INT;            // Variable of type INT
    in_String_Length : INT;         // Variable of type INT
    Iin_Y : INT;                    // Input variable of type INT
    Iin_X : INT;                    // Input variable of type INT
    Iby_ATTR : BYTE;                // Input variable of type BYTE
    Ist_STRING : STRING;            // Input variable of type STRING
    Xus_TN_SCREEN : us_TN_SCREEN;   // InOut variable of derived type us_TN_SCREEN
END_VAR
in_String_Length := LEN(Ist_STRING);
IF TN_SC_XY_ERROR(Iin_X + in_String_Length - 1,Iin_Y) THEN RETURN; END_IF;
Xus_TN_SCREEN.bya_Line_Update[Iin_Y] := TRUE; (* Zeile zum Updaten freigeben *)
in_Offset := (Iin_Y * 80) + Iin_X; (* Offset des ersten Zeichen berechnen *)
bo_Color := Iby_ATTR <> BYTE#00;

(* Zeichen und Attribute des String kopieren *)
FOR in_Char_Index := 1 TO in_String_Length DO
	IF bo_Color THEN Xus_TN_SCREEN.bya_COLOR[in_Offset] := Iby_ATTR; END_IF;
	Xus_TN_SCREEN.bya_CHAR[in_Offset] := OSCAT_BASIC.CODE(STR:=Ist_STRING,POS:=in_Char_Index);
	in_Offset := in_Offset + INT#1;
END_FOR;

Xus_TN_SCREEN.in_EOS_Offset := in_Offset; (* End of String Offset sichern *)

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
