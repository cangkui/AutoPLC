FUNCTION BYTE_OF_DWORD : BYTE
VAR_INPUT
    in : DWORD;
    N : BYTE;
END_VAR
BYTE_OF_DWORD := DWORD_TO_BYTE(SHR(in,SHL(n,3)));

(* revision history
hm	17. jan 2007	rev 1.0
	original version

hm	2. jan 2008		rev 1.1
	improved performance

hm	30. oct. 2008	rev 1.2
	improved performance
*)
END_FUNCTION
