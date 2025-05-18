FUNCTION TO_UML : string
VAR_INPUT
    IN : BYTE;
END_VAR
CASE in OF
	196:	(* Ä *)
		TO_UML := 'Ae';
	214:	(* Ö *)
		TO_UML := 'Oe';
	220:	(* Ü *)
		TO_UML := 'Ue';
	223:	(* ß *)
		TO_UML := 'ss';
	228:	(* ä *)
		TO_UML := 'ae';
	246:	(* ö *)
		TO_UML := 'oe';
	252:	(* ü *)
		TO_UML := 'ue';
ELSE
	TO_UML := CHR_TO_STRING(IN);
END_CASE;



(* revision history
hm	29. mar. 2008	rev 1.0
	original version

hm	17. dec. 2008	rev 1.1
	changes name of function chr to chr_to_string

*)
END_FUNCTION
