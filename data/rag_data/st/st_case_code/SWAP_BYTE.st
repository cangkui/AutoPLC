FUNCTION SWAP_BYTE : WORD
VAR_INPUT
    IN : WORD;
END_VAR
SWAP_BYTE := ROL(in,8);


(* revision history
hm		4. feb 2008	rev 1.0
	original version

*)
END_FUNCTION
