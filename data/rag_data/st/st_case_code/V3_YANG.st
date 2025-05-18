FUNCTION V3_YANG : REAL
VAR_INPUT
    A : Vector_3; (* Derived type as specified in XML *)
END_VAR

VAR
    la : REAL; (* Local variable of type REAL *)
END_VAR
la := V3_ABS(a);
IF la > 0.0 THEN
	V3_YANG := ACOS(A.Y / la);
END_IF;

(* revision history
hm	11 dec 2007	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	changed syntax of real constants to 0.0

*)
END_FUNCTION
