FUNCTION_BLOCK TIMER_2
VAR_INPUT
    DT_in : DATE;
    start : TIME_OF_DAY;
    duration : TIME;
    mode : BYTE;
    HOLIDAY : BOOL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    last_check : TIME;
    dat : DATE;
    daytime : TIME_OF_DAY;
    activation : TIME;
    tx : TIME;
    enabled : BOOL;
    init : BOOL;
    wday : INT;
END_VAR

VAR RETAIN
    run_date : DATE;
END_VAR
(* read system time *)
tx := DWORD_TO_TIME(OSCAT_BASIC.T_PLC_MS());

IF NOT init THEN
	init := TRUE;
	last_check := tx - t#100ms;
END_IF;

(* for performance resons exit if execution is not necessary *)
IF tx - last_check < T#100ms THEN RETURN; END_IF;

(* calculate temp variables *)
dat := DT_TO_DATE(dt_in);
daytime := DT_TO_TOD(dt_in);
wday := OSCAT_BASIC.DAY_OF_WEEK(dat);

(* calculate if any action is necessary today *)
CASE mode OF
	1..7:	(* mondays to fridays *)
		enabled := wday = BYTE_TO_INT(mode);
	11:	(* every day *)
		enabled := TRUE;
	12..16:	(* every nth day *)
		enabled := (DATE_TO_DWORD(dat)/86400) MOD (mode - 10) = 0;
	20:	(* monday - friday *)
		enabled := wday <= 5;
	21:	(* saturday and sunday *)
		enabled := wday > 5;
	22: (* workdays only *)
		enabled := wday <= 5 AND NOT holiday;
	23:	(* holidays and weekends *)
		enabled := wday > 5 OR holiday;
	24:	(* holidays only *)
		enabled := holiday;
	25:	(* 1st day of the month *)
		enabled := OSCAT_BASIC.DAY_OF_MONTH(dat) = 1;
	26:	(* last day of month *)
		enabled := OSCAT_BASIC.DAY_OF_MONTH(dat + T#1D) = 1;
	27:	(* 31st of december *)
		enabled := OSCAT_BASIC.DAY_OF_MONTH(dat) = 31 AND OSCAT_BASIC.MONTH_OF_DATE(dat) = 12;
	28:	(* 1st of january *)
		enabled := OSCAT_BASIC.DAY_OF_YEAR(dat) = 1;
ELSE
	enabled := FALSE;
END_CASE;

(* set output if necessary *)
IF enabled AND NOT Q AND daytime >= start AND run_date <> dat THEN
	Q := TRUE;
	activation := tx;
	run_date := dat;
ELSIF Q AND tx - activation >= duration THEN
	Q := FALSE;
END_IF;
last_check := Tx;


(* revision history
hm 22.2.2007		rev 1.1
	changed VAR RETAIN PERSISTENT to VAR RETAIN for better compatibility
	wago lon contollers do not support persisitent

hm	15.9.2007		rev 1.2
	replaced Time() with T_PLC_MS for compatibility and performance reasons

hm	7. oct. 2008	rev 1.3
	changed function month to month_of_date

hm	20. oct. 2008	rev 1.4
	improved performance
	deleted unnecessary trunc

hm	22. dec. 2008	rev 1.5
	make sure timer_2 is executed in the first cycle

*)
END_FUNCTION_BLOCK
