FUNCTION RES_SI : REAL
VAR_INPUT
    T : REAL;
    RS : REAL;
    TS : REAL;
END_VAR

VAR_CONSTANT
    A : REAL := 0.00764;
    B : REAL := 1.66E-05;
END_VAR

VAR
    TX : REAL;
END_VAR
TX := T - TS;
RES_SI := RS * ( 1.0 + A * TX + B * TX * TX);

(* revision history
hm	4. aug 2006	rev 1.0
	original version

hm	2. dec 2007	rev 1.1
	changed code for better performance

hm	11. mar. 2009	rev 1.2
	changed real constants to use dot syntax

*)
END_FUNCTION
