FUNCTION_BLOCK SCHEDULER_2
VAR_INPUT
    E0 : BOOL;
    E1 : BOOL;
    E2 : BOOL;
    E3 : BOOL;
    C0 : UINT;
    C1 : UINT;
    C2 : UINT;
    C3 : UINT;
    O0 : UINT;
    O1 : UINT;
    O2 : UINT;
    O3 : UINT;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
END_VAR

VAR
    sx : UINT;
END_VAR
Q0 := E0 AND (sx MOD C0 - O0 = 0);
Q1 := E1 AND (sx MOD C1 - O1 = 0);
Q2 := E2 AND (sx MOD C2 - O2 = 0);
Q3 := E3 AND (sx MOD C3 - O3 = 0);

(* increment cycle counter every cycle *)
sx := sx + 1;


(* revision history
hm 29. sep. 2008		rev 1.0
	original version


*)
END_FUNCTION_BLOCK
