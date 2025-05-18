FUNCTION_BLOCK SW_RECONFIG
VAR_INPUT
    IN : BOOL;
    TD : TIME;
    TR : TIME;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    T1 : OSCAT_BASIC.TON;
    T2 : OSCAT_BASIC.TON;
    INV : BOOL;
END_VAR
(* run debounce circuit *)
t1(in := IN, PT := TD);

IF tr > t#0s THEN
	(* generate output *)
	Q := t1.Q XOR inv;
	(* reconfiguration timer *)
	T2(in := Q, pt := TR);
	(* when T2.Q goes high then reverse INV *)
	IF T2.Q THEN INV := NOT INV; END_IF;
ELSE
	Q := t1.Q;
END_IF;



(* revision history
hm	22.10.2008	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
