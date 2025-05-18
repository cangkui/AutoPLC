FUNCTION_BLOCK TREND
VAR_INPUT
    X : REAL;  // Input variable
END_VAR

VAR_OUTPUT
    Q : BOOL;  // Output variable
    TU : BOOL; // Output variable
    TD : BOOL; // Output variable
    D : REAL;  // Output variable
END_VAR

VAR
    last_X : REAL;  // Local variable
END_VAR
TU := X > last_X;
TD := X < last_X;
Q := TU OR TD;
D := X - LAST_X;

last_X := X;



(* revision history
hm	21. aug. 2009	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
