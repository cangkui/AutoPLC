FUNCTION BIT_COUNT : INT
VAR_INPUT
    IN : DWORD;
END_VAR
WHILE in > 0 DO
	IF in.0 THEN Bit_Count := Bit_Count + 1; END_IF;
	in := SHR(in,1);
END_WHILE;



(* revision history
5.7.2007	rev 1.0		original version

10.9.2007	rev 1.1		hm
	changed algorithm for better performace
	the execution time has reduced by a factor of 5
	deleted unused variable temp
*)
END_FUNCTION
