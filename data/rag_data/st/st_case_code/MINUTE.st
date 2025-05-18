FUNCTION MINUTE : INT
VAR_INPUT
    itod : TOD;
END_VAR
MINUTE := DWORD_TO_INT(TOD_TO_DWORD(itod) / 60000 - TOD_TO_DWORD(itod) / 3600000 * 60);


(* change history

2.10.2006 changes name of input to itod

*)
END_FUNCTION
