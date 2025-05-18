FUNCTION AOUT1 : DWORD
VAR_INPUT
    in : REAL;
    Bit_0 : INT; // input_constant attribute
    Bit_N : INT := 31; // input_constant attribute, initial value set
    sign : INT := 255; // input_constant attribute, initial value set
    low : REAL; // input_constant attribute
    high : REAL := 10.0; // initial value set, input_constant attribute
END_VAR

VAR CONSTANT
    ff : DWORD := 2#1; // initial value set
END_VAR

VAR
    sx : BOOL;
    in2 : REAL;
END_VAR
(* if sign bit is defined *)
IF sign < 32 THEN
	sx := SIGN_R(in);
	in2 := ABS(in);
ELSE
	in2 := in;
END_IF;

(* begrenze in auf die zulässigen werte *)
in2 := LIMIT(low, in2, high);

(* Berechne den Ausgangswert *)
AOUT1 := SHL(REAL_TO_DWORD((in2 - low) / (high - low) * DWORD_TO_REAL(SHL(ff,bit_n - Bit_0 + 1) -1)),Bit_0);
IF sx THEN AOUT1 := SHL(ff,sign) OR AOUT1; END_IF;


(* revision history
hm	23. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
