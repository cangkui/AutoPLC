FUNCTION IP4_CHECK : BOOL
VAR_INPUT
    NIP : DWORD;
    LIP : DWORD;
    SM  : DWORD;
END_VAR
IP4_CHECK := (NIP AND sm) = (LIP AND SM);


(* revision history
hm		4. mar 2008	rev 1.0
	original version

*)
END_FUNCTION
