FUNCTION MUX_R2 : REAL
VAR_INPUT
    IN0 : REAL;
    IN1 : REAL;
    A : BOOL;
END_VAR
MUX_R2 := SEL(A, IN0, IN1);


(* revision history
hm	19. jan. 2007	rev 1.0
	original version

hm	16. oct. 2008	rev 1.1
	improved performance


*)
END_FUNCTION
