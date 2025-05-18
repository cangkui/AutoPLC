FUNCTION IP4_TO_STRING : string
VAR_INPUT
    IP4 : DWORD; 
END_VAR

VAR
    tmp : STRING[3]; 
END_VAR
(* mask bits AND INSERT AT given position *)
tmp := BYTE_TO_STRING(DWORD_TO_BYTE(IP4));
IP4_TO_STRING := CONCAT('...', tmp);
tmp := BYTE_TO_STRING(DWORD_TO_BYTE(SHR(IP4,8)));
IP4_TO_STRING := INSERT(IP4_TO_STRING, tmp, 2);
tmp := BYTE_TO_STRING(DWORD_TO_BYTE(SHR(IP4,16)));
IP4_TO_STRING := INSERT(IP4_TO_STRING, tmp, 1);
tmp := BYTE_TO_STRING(DWORD_TO_BYTE(SHR(IP4,24)));
IP4_TO_STRING := CONCAT(tmp, IP4_TO_STRING);



(* revision history
hm	22. apr 2008	rev 1.0
	original version

hm	10. mar. 2009	rev 1.1
	improved code
*)
END_FUNCTION
