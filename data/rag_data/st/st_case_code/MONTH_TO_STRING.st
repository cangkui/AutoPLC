FUNCTION MONTH_TO_STRING : string
VAR_INPUT
    MTH : INT;
    LANG : INT;
    LX : INT;
END_VAR

VAR
    ly : INT;
END_VAR
IF LANG <= 0 THEN Ly := LANGUAGE.DEFAULT; ELSE ly := MIN(LANG, language.LMAX); END_IF;
IF MTH < 1 OR MTH > 12 THEN
	RETURN;
ELSIF LX = 0 THEN
	MONTH_TO_STRING := language.MONTHS[ly, MTH];
ELSIF Lx = 3 THEN
	MONTH_TO_STRING := language.MONTHS3[ly, MTH];
END_IF;


(* revisison history

hm	21. sep. 2008	rev 1.0
	original version

hm	19. oct. 2008	rev 1.1
	changed language setup constants

hm 16.  jul 2024   rev 1.2
       addaed boundary check to respect array limits
*)
END_FUNCTION
