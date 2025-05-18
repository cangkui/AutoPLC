FUNCTION DWORD_TO_STRF : string
VAR_INPUT
    IN : DWORD;
    N : INT;
END_VAR
(* limit N to max 20 characters *)
(* convert dword to string first and cut to length N *)
DWORD_TO_STRF := FIX(DWORD_TO_STRING(in),LIMIT(0,N,20),48,1);


(* revision history
hm	26. jan.2007	rev 1.0		
	original version

hm	29. mar. 2008	rev 1.1
	changed STRING to STRING(20)
	limit the output string to max 20 digits
*)
END_FUNCTION
