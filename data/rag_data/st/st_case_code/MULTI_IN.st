FUNCTION MULTI_IN : REAL
VAR_INPUT
    in_1 : REAL;
    in_2 : REAL;
    in_3 : REAL;
    default : REAL;
    in_min : REAL;
    in_max : REAL;
    mode : BYTE;
END_VAR

VAR
    count : INT;
    F1 : BOOL;
    F2 : BOOL;
    F3 : BOOL;
END_VAR
F1 :=  in_1 > in_min AND in_1 < in_max;
F2 :=  in_2 > in_min AND in_2 < in_max;
F3 :=  in_3 > in_min AND in_3 < in_max;

CASE mode OF
0:	count := 0;
	IF F1 THEN
		count := count + 1;
		MULTI_IN := in_1;
	ELSE
		MULTI_IN := 0.0;
	END_IF;
	IF F2 THEN
		count := count + 1;
		MULTI_IN := MULTI_IN + in_2;
	END_IF;
	IF F3 THEN
		count := count + 1;
		MULTI_IN := MULTI_IN + in_3;
	END_IF;
	MULTI_IN := SEL(count = 0, MULTI_IN / INT_TO_REAL(count), default);

1:	MULTI_IN := SEL(F1, default, IN_1);

2:	MULTI_IN := SEL(F2, default, IN_2);

3:	MULTI_IN := SEL(F3, default, IN_3);

4:	MULTI_IN := default;

5:	MULTI_IN := SEL(F1, in_max, IN_1);
	IF F2 AND in_2 < MULTI_IN THEN MULTI_IN := in_2; END_IF;
	IF F3 AND in_3 < MULTI_IN THEN MULTI_IN := in_3; END_IF;
	IF MULTI_IN = in_max THEN MULTI_IN := default; END_IF;

6:	MULTI_IN := SEL(F1, in_min, IN_1);
	IF F2 AND in_2 > MULTI_IN THEN MULTI_IN := in_2; END_IF;
	IF F3 AND in_3 > MULTI_IN THEN MULTI_IN := in_3; END_IF;
	IF MULTI_IN = in_min THEN MULTI_IN := default; END_IF;

7:	IF F1 AND F2 AND F3 THEN MULTI_IN := MID3(in_1, in_2, in_3);
	ELSIF F1 AND F2 THEN MULTI_IN := MIN(in_1, in_2);
	ELSIF F1 AND F3 THEN MULTI_IN := MIN(in_1, in_3);
	ELSIF F2 AND F3 THEN MULTI_IN := MIN(in_2, in_3);
	ELSIF F1 THEN MULTI_IN := in_1;
	ELSIF F2 THEN MULTI_IN := in_2;
	ELSIF F3 THEN MULTI_IN := in_3;
	ELSE MULTI_IN := default;
	END_IF;

END_CASE;

(*
hm 1.1.2007		rev 1.1
	changed midr to mid3 function

hm	14. 10. 2008	rev 1.2
	corrected an error for in_3 overrange detection
	improved performance

hm	11. mar. 2009	rev 1.3
	changed real constants to use dot syntax

hm	18. jul. 2009	rev 1.4
	improved performance
*)
END_FUNCTION
