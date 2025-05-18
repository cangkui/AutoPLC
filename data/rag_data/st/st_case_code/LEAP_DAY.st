FUNCTION LEAP_DAY : BOOL
VAR_INPUT
    IDATE : DATE;
END_VAR
LEAP_DAY := DATE_TO_UDINT(IDATE) MOD 126230400 = 68169600;


(* change history

hm 	15. jun. 2008	rev 1.0
	original version

hm	7. oct. 2008	rev 1.1
	changed function month to month_of_date

hm	24. jan. 2011	rev 1.2
	improved performance
*)
END_FUNCTION
