FUNCTION_BLOCK TN_SC_WRITE_EOS
VAR
    TN_SC_WRITE: TN_SC_WRITE; // Derived type
    Iby_ATTR: BYTE;           // Input variable of type BYTE
    Ist_STRING: STRING;       // Input variable of type STRING
    Xus_TN_SCREEN: us_TN_SCREEN; // InOut variable of derived type
END_VAR
TN_SC_WRITE(Iin_Y := Xus_TN_SCREEN.in_EOS_Offset / 80, (* aktuelle Zeile ausrechnen *)
			Iin_X := Xus_TN_SCREEN.in_EOS_Offset MOD 80, (* aktuelle Spalte ausrechnen *)
			Iby_ATTR := Iby_ATTR,
			Ist_STRING := Ist_STRING,
			Xus_TN_SCREEN := Xus_TN_SCREEN);

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
