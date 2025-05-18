FUNCTION_BLOCK DLOG_CRON_TAB
VAR_INPUT
    ACTIVATE : BOOL;
    DTI : DT;
    SECOND : STRING[20];
    MINUTE : STRING[20];
    HOUR : STRING[20];
    DAY_OF_MONTH : STRING[20];
    DAY_OF_WEEK : STRING[20];
    MONTH : STRING[20];
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    CD : DLOG_CRON_DATA;
    activate_old : BOOL;
    datei : DATE;
    dti_last : DT;
    cnt1 : INT;
    cnt2 : INT;
    idx1 : INT;
    idx2 : INT;
    idx3 : INT;
    idx4 : INT;
    pos1 : INT;
    pos2 : INT;
    char_cnt : INT;
    trig_day : BOOL;
    trig_weekday : BOOL;
    str1 : STRING[OSCAT_BASIC.STRING_LENGTH];
    str2 : STRING[OSCAT_BASIC.STRING_LENGTH];
    str3 : STRING[OSCAT_BASIC.STRING_LENGTH];
    num_str1 : STRING[10];
END_VAR
IF ACTIVATE THEN
	IF NOT activate_old THEN
		CD.CE[0].VALUE_MIN := 0;	(* Sekunden *)
		CD.CE[0].VALUE_MAX := 59;
		CD.CE[1].VALUE_MIN := 0;	(* Minuten *)
		CD.CE[1].VALUE_MAX := 59;
		CD.CE[2].VALUE_MIN := 0;	(* Stunden *)
		CD.CE[2].VALUE_MAX := 59;
		CD.CE[3].VALUE_MIN := 1;	(* Tage *)
		CD.CE[3].VALUE_MAX := 31;
		CD.CE[4].VALUE_MIN := 1;	(* Wochentag *)
		CD.CE[4].VALUE_MAX := 7;
		CD.CE[5].VALUE_MIN := 1;	(* Monat *)
		CD.CE[5].VALUE_MAX := 12;

		FOR idx1 := 0 TO 5 DO
			idx3 := CD.CE[idx1].VALUE_MIN;
			idx4 := CD.CE[idx1].VALUE_MAx;
			FOR idx2 := idx3 TO idx4 DO
				CD.CE[idx1].ELEMENTS[idx2] := FALSE;
			END_FOR;
			CD.CE[idx1].ALL_SELECTED := FALSE;
		END_FOR;

		str1 := CONCAT(SECOND,'#');
		str1 := CONCAT(str1,MINUTE);
		str1 := CONCAT(str1,'#');
		str1 := CONCAT(str1,HOUR);
		str1 := CONCAT(str1,'#');
		str1 := CONCAT(str1,DAY_OF_MONTH);
		str1 := CONCAT(str1,'#');
		str1 := CONCAT(str1,DAY_OF_WEEK);
		str1 := CONCAT(str1,'#');
		str1 := CONCAT(str1,MONTH);

		FOR idx1 := 0 TO 5 DO
			str2 := ELEMENT_GET(BYTE#35, idx1, str1);
			IF LEN(str2) > 0 THEN
				cnt1 := ELEMENT_COUNT(BYTE#44, str2);	(* Trenzeichen ',' *)
			ELSE
				cnt1 := 1;
			END_IF;

			FOR idx2 := 0 TO cnt1 -1 DO
				str3 := ELEMENT_GET(BYTE#44, idx2, str2);
				IF FIND(str3, '*/') > 0 AND LEN(str3) <= 10 THEN
					pos1 := OSCAT_BASIC.DEC_TO_INT(str3);
					IF pos1 > 0 THEN
						idx3 := CD.CE[idx1].VALUE_MIN;
						idx4 := CD.CE[idx1].VALUE_MAX;
						WHILE idx3 <= idx4 DO
							CD.CE[idx1].ELEMENTS[idx3] := TRUE;
							idx3 := idx3 + pos1;
						END_WHILE;
					END_IF;

				ELSE
					IF FIND(str3, '*') > 0 OR LEN(str3) = 0 THEN
						pos1 := CD.CE[idx1].VALUE_MIN;
						pos2 := CD.CE[idx1].VALUE_MAX;
						CD.CE[idx1].ALL_SELECTED := TRUE;
					ELSE
						cnt2 := ELEMENT_COUNT(BYTE#45, str3);	(* Trenzeichen '-' *)
						char_cnt := LEN(str3);
						IF char_cnt >= 1 AND char_cnt <= 10 THEN
							num_str1 := ELEMENT_GET(BYTE#45, 0, str3);
							pos1 := OSCAT_BASIC.DEC_TO_INT(num_str1);
							pos2 := pos1;
							IF cnt2 = 2 THEN
								num_str1 := ELEMENT_GET(BYTE#45, 1, str3);
								pos2 := OSCAT_BASIC.DEC_TO_INT(num_str1);
							END_IF;
						END_IF;
					END_IF;

					IF pos1 >= CD.CE[idx1].VALUE_MIN AND pos2 <= CD.CE[idx1].VALUE_MAX THEN
						FOR idx3 := pos1 TO pos2 DO
							CD.CE[idx1].ELEMENTS[idx3] := TRUE;
						END_FOR;
					END_IF;
				END_IF;
			END_FOR;
		END_FOR;
	ELSIF DTI <> dti_last THEN
		CD.CE[0].VALUE := OSCAT_BASIC.SECOND_OF_DT(DTI); (* Sekunden *)
		CD.CE[1].VALUE := OSCAT_BASIC.MINUTE_OF_DT(DTI); (* Minuten *)
		CD.CE[2].VALUE := OSCAT_BASIC.HOUR_OF_DT(DTI); (* Stunden *)
		datei := DT_TO_DATE(DTI);
		CD.CE[3].VALUE := OSCAT_BASIC.DAY_OF_MONTH(datei); (* Tag *)
		CD.CE[4].VALUE := OSCAT_BASIC.DAY_OF_WEEK(datei); (* Wochentag *)
		CD.CE[5].VALUE := OSCAT_BASIC.MONTH_OF_DATE(datei); (*  Monat *)
		dti_last := DTI;

		idx2 := CD.CE[0].VALUE;	(* sekunden *)
		Q := CD.CE[0].ELEMENTS[idx2] = TRUE;
		idx2 := CD.CE[1].VALUE;	(* minuten *)
		Q := Q AND CD.CE[1].ELEMENTS[idx2] = TRUE;
		idx2 := CD.CE[2].VALUE;	(* stunden *)
		Q := Q AND CD.CE[2].ELEMENTS[idx2] = TRUE;
		idx2 := CD.CE[5].VALUE;	(* monat *)
		Q := Q AND CD.CE[5].ELEMENTS[idx2] = TRUE;

		(* auswertung Tag und Wochentag *)
		idx2 := CD.CE[3].VALUE;	(* Tag *)
		trig_day:= CD.CE[3].ELEMENTS[idx2] = TRUE;
		idx2 := CD.CE[4].VALUE;	(* Wochentag *)
		trig_weekday := CD.CE[4].ELEMENTS[idx2] = TRUE;

		IF CD.CE[3].ALL_SELECTED = FALSE AND CD.CE[4].ALL_SELECTED = TRUE THEN
			Q := Q AND trig_day; (* nur Tag auswerten *)
		ELSIF CD.CE[3].ALL_SELECTED = TRUE AND CD.CE[4].ALL_SELECTED = FALSE THEN
			Q := Q AND trig_weekday; (* nur Wochentag auswerten *)
		ELSIF CD.CE[3].ALL_SELECTED = FALSE AND CD.CE[4].ALL_SELECTED = FALSE THEN
			Q := Q AND (trig_weekday OR trig_day); (* Tag und Wochentag auswerten *)
		END_IF;
	END_IF;

END_IF;
activate_old := ACTIVATE;

(* revision history
ks	27. oct. 2011	rev 1.0
	original version
*)
END_FUNCTION_BLOCK
