FUNCTION V3_ABS : REAL
VAR_INPUT
    A : ARRAY[1..3] OF REAL;  // Assuming Vector_3 is represented as an array of 3 REAL values
END_VAR
V3_ABS := SQRT(A.X * A.X + A.Y * A.Y + A.Z * A.Z);


(* revision history
hm		11 dec 2007	rev 1.0
	original version

*)
END_FUNCTION
