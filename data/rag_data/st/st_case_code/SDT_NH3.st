FUNCTION SDT_NH3 : REAL
VAR_INPUT
    P : REAL;
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


IF P < 1.0E-3 THEN
	SDT_NH3 := -110.0;
ELSIF P < 1.0 THEN
	SDT_NH3 := 506.713 / (3.18757 - LOG(P)) - 192.37 ;
ELSE
	SDT_NH3 := 1113.928 / (4.86886 - LOG(P)) - 262.71 ;
END_IF;



(* revision history

hm 20 aug 2009		rev 1.0
	original version

*)
END_FUNCTION
