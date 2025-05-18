FUNCTION _BUFFER_CLEAR : BOOL
VAR_INPUT
    PT : POINTER TO BYTE; // Pointer to BYTE
    SIZE : UINT;         // Unsigned Integer
END_VAR

VAR
    ptw : POINTER TO DWORD; // Pointer to DWORD
    temp : DWORD;          // DWORD variable
    end : DWORD;           // DWORD variable
    end32 : DWORD;         // DWORD variable
END_VAR
(* this routine uses 32 bit access to gain speed *)
(* first access bytes till pointer is aligned for 32 bit access *)
temp := pt;
end := temp + UINT_TO_DWORD(size);
end32 := end - 3;
WHILE (pt < end) AND ((temp AND 16#00000003) > 0) DO
	pt^ := 0;
	pt := pt + 1;
	temp := temp + 1;
END_WHILE;
(* pointer is aligned, now copy 32 bits at a time *)
ptw := pt;
WHILE ptw < end32 DO (* *)
	ptw^ := 0;
	ptw := ptw + 4;
END_WHILE;
(* copy the remaining bytes in byte mode *)
pt := ptw;
WHILE pt < end DO
	pt^ := 0;
	pt := pt + 1;
END_WHILE;

_BUFFER_CLEAR := TRUE;

(* revision History
hm 	5. mar. 2008	rev 1.0
	original version

hm	16. mar. 2008	rev 1.1
	added type conversion to avoid warnings under codesys 30
	chaged type of input size to uint
	deleted unused variable i

hm	31. oct. 2008	rev 1.2
	corrected an error while routine would write outside of arrays

*)
END_FUNCTION
