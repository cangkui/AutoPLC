FUNCTION MUX_2 : BOOL
VAR_INPUT
    D0 : BOOL;
    D1 : BOOL;
    A0 : BOOL;
END_VAR
MUX_2 := SEL(A0, D0, D1);


(*
revision history:
hm	5.10.2006		rev 1.1
	changed to ST Langage for better portability.

hm	16. oct. 2008	rev 1.2
	improved performance

*)
END_FUNCTION
