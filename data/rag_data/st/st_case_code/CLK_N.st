FUNCTION_BLOCK CLK_N
VAR_INPUT
    N : INT;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    edge : BOOL;
    stime : DWORD;
    clk : BOOL;
END_VAR
stime := SHR(T_PLC_MS(),N);
clk := stime.0;
Q := clk XOR edge;
edge := CLK;

(* revision history
hm	16. dec 2007		rev 1.0
	original version

*)
END_FUNCTION_BLOCK
