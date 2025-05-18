FUNCTION SECOND_OF_DT : INT
VAR_INPUT
    XDT : DT;
END_VAR
SECOND_OF_DT := DWORD_TO_INT(DT_TO_DWORD(XDT) MOD 60);

(* revision history
hm		6.9.2008	rev 1.0
	original version

*)
END_FUNCTION
