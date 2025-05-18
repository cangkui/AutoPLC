FUNCTION TIMER_EVENT_DECODE : OSCAT_BASIC.TIMER_EVENT
VAR_INPUT
    EVENT : STRING(OSCAT_BASIC.STRING_LENGTH);
    LANG : INT;
END_VAR

VAR
    pos : INT;
    start : INT;
    stop : INT;
    pt : POINTER TO ARRAY[1..255] OF BYTE;
    step : INT;
    tmp : STRING(60);
END_VAR
(* a timer event is specified as follows:
	$typ;ch;day;start;duration;land;lor#
$ 		= start of definition
typ 	= type of definition	(number)
channel	= channel number		(number)
day 	= day number			(number)
start	= start time			(time of day)
duration= duration of event		(time)
land	= mask for output AND	(8bit value)
lor		= mask for ourput OR	(8bit value)
#		= end of definition
*)

(* check for start and ending character *)
IF (OSCAT_BASIC.LEFT(event,1) <> '<') AND (OSCAT_BASIC.RIGHT(event,1) <> '>') THEN RETURN; END_IF;
stop := OSCAT_BASIC.LEN(event);
pt := ADR(event);
start := 2;

(* parse the string for next ; *)
FOR pos := 2 TO stop DO
	IF pt^[pos] = 59 OR pt^[pos] = 62 THEN
		tmp := OSCAT_BASIC.MID(event, pos - start, start);
		CASE step OF
			0:	(* read type *)
				TIMER_EVENT_DECODE.typ := OSCAT_BASIC.FSTRING_TO_BYTE(tmp);

			1:	(* read channel *)
				TIMER_EVENT_DECODE.channel := OSCAT_BASIC.FSTRING_TO_BYTE(tmp);

			2:	(* read day *)
				IF OSCAT_BASIC.IS_CC(tmp,'0123456789abcdefABCDEF#') THEN
					TIMER_EVENT_DECODE.day := OSCAT_BASIC.FSTRING_TO_BYTE(tmp);
				ELSIF TIMER_EVENT_DECODE.typ = 2 THEN
					TIMER_EVENT_DECODE.DAY := OSCAT_BASIC.FSTRING_TO_WEEK(tmp, LANG);
				ELSE
					TIMER_EVENT_DECODE.DAY := INT_TO_BYTE(OSCAT_BASIC.FSTRING_TO_WEEKDAY(tmp, LANG));
				END_IF;

			3:	(* read start *)
				TIMER_EVENT_DECODE.start := STRING_TO_TOD(tmp);

			4:	(* read duaration *)
				TIMER_EVENT_DECODE.duration := STRING_TO_TIME(tmp);

			5:	(* read land *)
				TIMER_EVENT_DECODE.land := OSCAT_BASIC.FSTRING_TO_BYTE(tmp);

			6:	(* read lor *)
				TIMER_EVENT_DECODE.LOR := OSCAT_BASIC.FSTRING_TO_BYTE(tmp);
		END_CASE;
		start := pos + 1;
		step := step + 1;
	END_IF;
END_FOR;




(* revision history
hm	17. jun 2008	rev 1.0
	original version

hm	25. oct. 2008	rev 1.1
	adding input lang for language selection
*)
END_FUNCTION
