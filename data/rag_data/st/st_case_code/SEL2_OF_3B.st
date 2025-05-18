FUNCTION_BLOCK SEL2_OF_3B
VAR_INPUT
    IN1 : BOOL;
    IN2 : BOOL;
    IN3 : BOOL;
    TD : TIME;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    W : BOOL;
END_VAR

VAR
    TDEL : TON;
END_VAR
Q := (IN1 AND IN2) OR (IN1 AND IN3) OR (IN2 AND IN3);
TDEL(IN := (in1 XOR in2) OR (in1 XOR in3) OR (in2 XOR in3), PT := TD);
W := TDEL.Q;


(* revision history
hm		19. may 2008	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
