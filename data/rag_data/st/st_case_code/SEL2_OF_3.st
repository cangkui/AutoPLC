FUNCTION_BLOCK SEL2_OF_3
VAR_INPUT
    IN1 : REAL;
    IN2 : REAL;
    IN3 : REAL;
    D : REAL;
END_VAR

VAR_OUTPUT
    Y : REAL;
    W : INT;
    E : BOOL;
END_VAR

VAR
    D12 : BOOL;
    D23 : BOOL;
    D31 : BOOL;
END_VAR
D12 := ABS(IN1-IN2) <= D;
D23 := ABS(IN2-IN3) <= D;
D31 := ABS(IN3-IN1) <= D;

IF (D12 AND D23) OR (D12 AND D31) OR (D23 AND D31) THEN
	(* all 3 inputs are valid *)
	Y := (IN1 + IN2 + IN3) * 0.333333333333;
	E := FALSE;
	W := 0;
ELSIF D12 THEN
	(* only inut 1 and 2 are valid *)
	Y := (In1 + IN2) * 0.5;
	E := FALSE;
	W := 3;
ELSIF D23 THEN
	(* only inut 2 and 3 are valid *)
	Y := (In2 + IN3) * 0.5;
	E := FALSE;
	W := 1;
ELSIF D31 THEN
	(* only inut 3 and 1 are valid *)
	Y := (In3 + IN1) * 0.5;
	E := FALSE;
	W := 2;
ELSE
	(* no calculation possible *)
	E := TRUE;
	W := 4;
END_IF;


(* revision history
hm	18. may 2008	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	improved code
*)
END_FUNCTION_BLOCK
