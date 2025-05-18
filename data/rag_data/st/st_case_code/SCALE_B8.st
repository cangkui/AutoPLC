FUNCTION SCALE_B8 : REAL
VAR_INPUT
    in1 : BYTE;
    in2 : BYTE;
    in3 : BYTE;
    in4 : BYTE;
    in5 : BYTE;
    in6 : BYTE;
    in7 : BYTE;
    in8 : BYTE;
    K : REAL;
    O : REAL;
    in1_min : REAL;
    in1_max : REAL := 1000.0;
    in2_min : REAL;
    in2_max : REAL := 1000.0;
    in3_min : REAL;
    in3_max : REAL := 1000.0;
    in4_min : REAL;
    in4_max : REAL := 1000.0;
    in5_min : REAL;
    in5_max : REAL := 1000.0;
    in6_min : REAL;
    in6_max : REAL := 1000.0;
    in7_min : REAL;
    in7_max : REAL := 1000.0;
    in8_min : REAL;
    in8_max : REAL := 1000.0;
END_VAR
SCALE_B8 := 	(((in1_max - in1_min)* in1 + (in2_max - in2_min)* in2 + (in3_max - in3_min)* in3 + (in4_max - in4_min)* in4 +
				(in5_max - in5_min)* in5 + (in6_max - in6_min)* in6 + (in7_max - in7_min)* in7 + (in8_max - in8_min)* in8) * 0.003921569
				 + in1_min + in2_min + in3_min + in4_min + in5_min + in6_min + in7_min + in8_min) * K + O;


(* revision History
hm	19. jan.2007	rev 1.1
	changed outputs to real to avoid overflow of integer
	added offset for better cascading of scale functions

hm	6. jan. 2008	rev 1.2
	improved performance

hm	26. oct. 2008	rev 1.3
	code optimization

hm	3. nov. 2008	rev 1.4
	used wrong factor in formula

*)
END_FUNCTION
