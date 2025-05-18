FUNCTION TN_SC_SHADOW_ATTR : BYTE
VAR_INPUT
    Iby_ATTR : BYTE;
END_VAR
(* Background Farbe auf Schwarz ändern      *)
(* und dunkle Fordergrundfarben einschalten *)  
TN_SC_SHADOW_ATTR := ((Iby_ATTR AND BYTE#2#11110000) OR BYTE#2#00001000);

(* revision history
ks	10. mai. 2010	rev 1.0
	original version
*)
END_FUNCTION
