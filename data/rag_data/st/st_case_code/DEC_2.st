FUNCTION_BLOCK DEC_2
VAR_INPUT
    D : BOOL;
    A : BOOL;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
END_VAR
Q0 := D AND NOT A;
Q1 := D AND A;

(* revision history
hm 3.3.2007	rev 1.1
	rewritten in ST for better compatibility

*)
END_FUNCTION_BLOCK
