FUNCTION JD2000 : REAL
VAR_INPUT
    DTI : TIME;
END_VAR
JD2000 := DWORD_TO_REAL(DT_TO_DWORD(DTI) - 946728000) / 86400.0;

(* revision histroy
hm	15. jul. 2008	rev 1.0
	original release


*)
END_FUNCTION
