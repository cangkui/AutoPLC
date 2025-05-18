FUNCTION SCALE_X2 : REAL
VAR_INPUT
    IN1 : BOOL;
    IN2 : BOOL;
    K : REAL;
    O : REAL;
    IN1_MIN : REAL;
    IN1_MAX : REAL := 1000.0;
    IN2_MIN : REAL;
    IN2_MAX : REAL := 1000.0;
END_VAR
SCALE_X2 := (SEL(IN1, IN1_MIN, IN1_MAX)+ SEL(IN2, IN2_MIN, IN2_MAX)) * k + o;


(* revision history
hm	19. jan, 2007	rev 1.0
	original version

hm	26. oct. 2008	rev 1.1
	code optimized

*)
END_FUNCTION
