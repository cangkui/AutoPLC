FUNCTION MONTH_END : DATE
VAR_INPUT
    IDATE : DATE;
END_VAR
MONTH_END := DWORD_TO_DATE(DATE_TO_DWORD(SET_DATE(YEAR_OF_DATE(idate),MONTH_OF_DATE(idate)+1,1)) - 86400);



(* revision history
hm	15. jun. 2008	rev 1.0
	original version	

hm	7. oct. 2008	rev 1.1
	changed function year to year_of_date
	changed function month to month_of_date

*)
END_FUNCTION
