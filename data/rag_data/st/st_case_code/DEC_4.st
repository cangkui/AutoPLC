FUNCTION_BLOCK DEC_4
VAR_INPUT
    D : BOOL;
    A0 : BOOL;
    A1 : BOOL;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
END_VAR
Q0 := D AND NOT A0 AND NOT A1;
Q1 := D AND A0 AND NOT A1;
Q2 := D AND NOT A0 AND A1;
Q3 := D AND A0 AND A1;

(* revision history
hm 3.3.2007	rev 1.1
	rewritten in ST for better compatibility

*)
END_FUNCTION_BLOCK
