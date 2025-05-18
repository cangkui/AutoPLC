FUNCTION HOUR : INT
VAR_INPUT
    itod : TIME_OF_DAY; // Declaration of input variable itod of type TIME_OF_DAY
END_VAR
HOUR := DWORD_TO_INT(TOD_TO_DWORD(itod) / 3600000);


(* change history
hm 4. aug 2006	rev 1.0
	original version

hm 2.10.2006 	rev 1.1
	changed name of input to itod

*)
END_FUNCTION
