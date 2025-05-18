FUNCTION SHL1 : DWORD
VAR_INPUT
    IN : DWORD;
    N : INT;
END_VAR

VAR_CONSTANT
    temp : DWORD := 16#FFFFFFFF;
END_VAR
SHL1 := SHR(temp,32-N) OR SHL(IN,N);


(* revision history
hm	14.9.2007		rev 1.0
	original version

hm	27. dec 2007	rev 1.1
	changed code for better performance

*)
END_FUNCTION
