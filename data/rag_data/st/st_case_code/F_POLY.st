FUNCTION F_POLY : REAL
VAR_INPUT
    X : REAL;
    C : ARRAY[0..7] OF REAL;
END_VAR
F_POLY := ((((((( c[7] * x + c[6] ) * x + c[5] ) * x + c[4] ) * x + c[3] ) * x + c[2] ) * x + c[1] ) * x + c[0] ) ;


(* revision history
hm		20. may. 2008		rev 1.0
	original version

hm	18. mar. 2011	rev 1.1
	improved performance
*)
END_FUNCTION
