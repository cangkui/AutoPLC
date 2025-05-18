FUNCTION SDD_NH3 : REAL
VAR_INPUT
    T : REAL;
END_VAR
(*
calculate according to NIST
Antoine Equation Parameters
log10(P) = A - (B / (T + C))
	P = vapor pressure (bar)
	T = temperature (K)
Temperature (K)	A	B	C	Reference	Comment

 T = 164.0 - 239.5, A =	3.18757, B =506.713, C=-80.78	Stull, 1947	Coefficents calculated by NIST from author's data.
 T = 239.5 - 371.4, A =	4.86886, B = 1113.928, C = -10.409	Stull, 1947	Coefficents calculated by NIST from author's data.
*)



IF T < -33.65 THEN
	SDD_NH3 := EXP(7.3396511649 - ( 1166.7498002 / ( T + 192.37 )));
ELSE
	SDD_NH3 := EXP(11.210964456 - ( 2564.9140075 / ( T + 262.741 )));
END_IF;



(* revision history

hm 19 aug 2009		rev 1.0
	original version

*)
END_FUNCTION
