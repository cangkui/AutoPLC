FUNCTION AOUT : DWORD
VAR_INPUT
    in : REAL;
    Bits : BYTE;
    sign : BYTE := 255;
    low : REAL;
    high : REAL := 10.0;
END_VAR_INPUT

VAR CONSTANT
    ff : DWORD := 2#1;
END_VAR_CONSTANT

VAR
    in2 : REAL;
    sx : BOOL;
END_VAR
(* if sign bit is defined *)
IF sign < 32 THEN
	sx := sign_R(in);
	in2 := ABS(in);
ELSE
	in2 := in;
END_IF;

(* begrenze in auf die zulässigen werte *)
in2 := LIMIT(low, in2, high);

(* Berechne den Ausgangswert *)
Aout := REAL_TO_DWORD((in2 - low) / (high - low) * DWORD_TO_REAL(SHL(ff,bits) -1));
IF sx THEN Aout := SHL(ff,sign) OR Aout; END_IF;

(*
revision history

hm 18.1.2007		rev 1.1
	renamed Modul to aout.
	changed Output to 32 Bit max.
	corrected error with sign bit.

hm	13.9.2007		rev 1.2
	changed code to avoid warning under codesys 2.8.1

hm	2. dec 2007		rev 1.3
	changed code for better performance

hm	23. feb 2008	rev 1.4
	changed code for better performance
*)
END_FUNCTION
