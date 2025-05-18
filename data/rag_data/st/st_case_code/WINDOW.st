FUNCTION WINDOW : BOOL
VAR_INPUT
    low : REAL;
    in : REAL;
    high : REAL;
END_VAR
WINDOW := in < high AND in > low;


(* revision history
hm	22.1.2007	rev 1.1
	changed the order of inputs to low, in, high

*)
END_FUNCTION
