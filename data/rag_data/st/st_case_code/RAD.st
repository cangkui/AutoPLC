FUNCTION RAD : REAL
VAR_INPUT
    DEG : REAL; // Degrees as input variable
END_VAR
RAD := MODR(0.0174532925199433 * DEG, math.PI2);

(* revision history
hm	4. aug 2006		rev 1.0
	original version

hm 	16. oct 2007	rev 1.1
	added modr statement which prohibits rad to become bigger than 2PI

hm	18. oct 2008	rev 1.2
	using math constants

*)
END_FUNCTION
