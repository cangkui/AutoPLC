FUNCTION SECOND : REAL
VAR_INPUT
    itod : TOD;
END_VAR
SECOND := DWORD_TO_REAL(TOD_TO_DWORD(itod) - TOD_TO_DWORD(itod)/60000 * 60000) / 1000.0;



(* change history

hm	2. oct. 2006 rev 1.1 
	changed name of input to itod

*)
END_FUNCTION
