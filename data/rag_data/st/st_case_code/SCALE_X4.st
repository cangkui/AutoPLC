FUNCTION SCALE_X4 : REAL
VAR_INPUT
    IN1 : BOOL;
    IN2 : BOOL;
    IN3 : BOOL;
    IN4 : BOOL;
    K : REAL;
    O : REAL;
    IN1_MIN : REAL; // input_constant
    IN1_MAX : REAL := 1000.0; // input_constant
    IN2_MIN : REAL; // input_constant
    IN2_MAX : REAL := 1000.0; // input_constant
    IN3_MIN : REAL; // input_constant
    IN3_MAX : REAL := 1000.0; // input_constant
    IN4_MIN : REAL; // input_constant
    IN4_MAX : REAL := 1000.0; // input_constant
END_VAR
SCALE_X4 := (SEL(IN1,IN1_MIN, IN1_MAX) + SEL(IN2, IN2_MIN, IN2_MAX)+SEL(IN3, IN3_MIN, IN3_MAX)+ SEL(IN4, IN4_MIN, IN4_MAX)) * k + o;


(* revision history
hm	19. jan. 2008	rev 1.0
	original version

hm	26. oct. 2008	rev 1.1
	optimized code
*)
END_FUNCTION
