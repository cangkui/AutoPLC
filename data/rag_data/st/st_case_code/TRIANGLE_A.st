FUNCTION TRIANGLE_A : REAL
VAR_INPUT
    S1 : REAL;
    A : REAL;
    S2 : REAL;
    S3 : REAL;
END_VAR
IF A = 0.0 THEN
	TRIANGLE_A := SQRT((s1+s2+S3) * (s1+s2-S3) * (S2+S3-S1) * (S3+S1-S2)) * 0.25;
ELSE
	TRIANGLE_A := S1 * S2 * SIN(RAD(A)) * 0.5;
END_IF;

(* revision histroy
hm	16. oct 2007	rev 1.0
	original version

hm	22. feb 2008	rev 1.1
	improved performance

hm	10. mar. 2009	rev 1.2
	changed syntax of real constants to 0.0

*)
END_FUNCTION
