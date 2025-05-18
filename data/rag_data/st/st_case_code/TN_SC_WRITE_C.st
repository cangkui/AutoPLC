FUNCTION_BLOCK TN_SC_WRITE_C
VAR
    FB_TN_SC_WRITE: TN_SC_WRITE;    // Local variable
    Iin_Y: INT;                      // Input variable
    Iin_X: INT;                      // Input variable
    Iby_ATTR: BYTE;                  // Input variable
    Ist_STRING: STRING;              // Input variable
    Iin_LENGTH: INT;                 // Input variable
    Iin_OPTION: INT;                 // Input variable
    Xus_TN_SCREEN: us_TN_SCREEN;     // InOut variable
END_VAR
(* Text ausgeben *)
FB_TN_SC_WRITE( Iin_Y := Iin_Y,
				Iin_X := Iin_X,
				Iby_ATTR := Iby_ATTR,
				Ist_STRING := OSCAT_BASIC.FIX(str:=Ist_STRING,L:=Iin_LENGTH,C:=BYTE#32,M:=Iin_OPTION),
				Xus_TN_SCREEN := Xus_TN_SCREEN);

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
