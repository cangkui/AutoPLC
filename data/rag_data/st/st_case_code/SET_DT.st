FUNCTION SET_DT : DT
VAR_INPUT
    year : INT;
    month : INT;
    day : INT;
    hour : INT;
    minute : INT;
    second : INT;
END_VAR
SET_DT := DWORD_TO_DT(DATE_TO_DWORD(SET_DATE(YEAR, MONTH, day)) + INT_TO_DWORD(SECOND) + INT_TO_DWORD(MINUTE) * 60 + INT_TO_DWORD(HOUR) * 3600);


(* revision history
hm	4. aug. 2006		rev 1.0
	original version

hm		19 sep. 2007	rev 1.1
	use function leap_year to calculate leap year, more exceptions are handled

hm		1. okt 2007		rev 1.2
	added step7 compatibility
	call function set_date

hm		8. oct 2007		rev 1.3
	deleted unused variables count and leap

hm		1. 11 2007		rev 1.4
	converted hour type integer to dword in calculation to avoid overrun on möller ecp4

hm		16. mar 2008	rev 1.5
	added type conversions to avoid warnings under codesys 3.0
*)
END_FUNCTION
