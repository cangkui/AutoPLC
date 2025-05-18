FUNCTION AIN : REAL
VAR_INPUT
    in : DWORD;
    Bits : BYTE;
    sign : BYTE := 255;
    low : REAL;
    high : REAL := 10.0;
END_VAR

VAR_CONSTANT
    ff : DWORD := 16#FFFFFFFF;
END_VAR

VAR
    temp1 : DWORD;
    temp2 : DWORD;
    sx : BOOL;
END_VAR
(* extract the sign bit *)
IF sign < 32 THEN
	temp1 := SHR(in,sign);
	sx := temp1.0;
ELSE
	sx := FALSE;
END_IF;
temp1 := SHR(ff, 32-bits);
temp2 := in AND temp1;
AIN := (high - low) * DWORD_TO_REAL(temp2) / DWORD_TO_REAL(temp1) + low;
IF sx THEN AIN := -AIN; END_IF;

(* revision history
hm 18.8.2006	rev 1.1
	fixed an error with low value negative and high value 0.

hm 19.1.2007	rev 1.2
	fixed an error with sign bit.

hm	13.9.2007	rev 1.3
	changed code to avoid warning under codesys 2.8.1

hm	2. dec 2007	rev 1.4
	changed code for better performance

hm	16. mar 2008	rev 1.5
	added type conversions to avoid warnings under codesys 30
*)
END_FUNCTION
