FUNCTION REFRACTION : REAL
VAR_INPUT
    ELEV : REAL; 
END_VAR
elev := LIMIT(-1.9, elev, 80.0);
REFRACTION := 0.0174532925199433 / TAN(0.0174532925199433 * (ELEV + 10.3 / (ELEV + 5.11)));


(* revision histroy
hm	14. jul. 2008	rev 1.0
	original release

hm	7. mar. 2009	rev 1.1
	using new formula

*)
END_FUNCTION
