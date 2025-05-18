FUNCTION SCALE_X8 : REAL
VAR_INPUT
    in1 : BOOL;
    in2 : BOOL;
    in3 : BOOL;
    in4 : BOOL;
    in5 : BOOL;
    in6 : BOOL;
    in7 : BOOL;
    in8 : BOOL;
    K : REAL;
    O : REAL;
    in1_min : REAL; // input_constant
    in1_max : REAL := 1000.0; // input_constant
    in2_min : REAL; // input_constant
    in2_max : REAL := 1000.0; // input_constant
    in3_min : REAL; // input_constant
    in3_max : REAL := 1000.0; // input_constant
    in4_min : REAL; // input_constant
    in4_max : REAL := 1000.0; // input_constant
    in5_min : REAL; // input_constant
    in5_max : REAL := 1000.0; // input_constant
    in6_min : REAL; // input_constant
    in6_max : REAL := 1000.0; // input_constant
    in7_min : REAL; // input_constant
    in7_max : REAL := 1000.0; // input_constant
    in8_min : REAL; // input_constant
    in8_max : REAL := 1000.0; // input_constant
END_VAR
SCALE_X8 := (SEL(IN1,IN1_MIN, IN1_MAX)+ SEL(IN2,IN2_MIN, IN2_MAX)+SEL(IN3,IN3_MIN, IN3_MAX)+ SEL(IN4,IN4_MIN, IN4_MAX)
				+SEL(IN5,IN5_MIN, IN5_MAX)+ SEL(IN6,IN6_MIN, IN6_MAX)+SEL(IN7,IN7_MIN, IN7_MAX)+ SEL(IN8,IN8_MIN, IN8_MAX)) * k + o;



(* revision history
hm	19. jan. 2008	rev 1.0
	original version

hm	26. oct. 2008	rev 1.1
	optimized code

hm	24. jan. 2008	rev 1.2
	corrected error in formula
*)
END_FUNCTION
