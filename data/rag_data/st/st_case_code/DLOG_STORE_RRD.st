FUNCTION_BLOCK DLOG_STORE_RRD
VAR_IN_OUT
    X : DLOG_DATA;    
END_VAR

VAR_INPUT
    ENABLE : BOOL;
    TRIG_M : BOOL;
    TRIG_T : TIME;
    URL : STRING[OSCAT_BASIC.STRING_LENGTH];
    DTI : DT;
    SEP : BYTE;
    DNS_IP4 : DWORD;
    TIMEOUT : TIME;
END_VAR

VAR_OUTPUT
    ERROR_C : DWORD;
    ERROR_T : BYTE;
END_VAR

VAR
    UCB : UNI_CIRCULAR_BUFFER;
    IP_C : IP_C; (* IP_Control Verwaltungsstruktur *)
    S_BUF : NETWORK_BUFFER;
    R_BUF : NETWORK_BUFFER;
    URL_DATA : url;
    DNS_CLIENT : DNS_CLIENT;
    HTTP_GET : HTTP_GET;
    IPC : IP_CONTROL;
    trig_auto : BOOL;
    trig_stored : BOOL;
    trig_tmp : BOOL;
    trig_m_last : BOOL;
    dti_last : DT := DATE_AND_TIME#2070-2-6-6:28:15;
    step_1 : INT;
    step_2 : INT;
    sep_char : STRING[1];
    url_query : STRING[OSCAT_BASIC.STRING_LENGTH];
    n : INT;
END_VAR
X.DTI := DTI; (* Datum / Zeit eintragen *)
X.CLOCK_TRIG := DTI <> dti_last;
dti_last := DTI;
trig_auto := FALSE;
IF X.CLOCK_TRIG THEN (* Impuls in jeder Sekunde *)
	IF TRIG_T >= T#1000ms THEN (* automatischer Trigger *)
		trig_auto := DT_TO_UDINT(DTI) MOD (TIME_TO_UDINT(TRIG_T) / UDINT#1000) = UDINT#0;
	END_IF;
END_IF;

CASE step_1 OF

00:	IF ENABLE THEN
		IF X.ID_MAX = USINT#0 THEN (* nur einmal nach Satellitenbausteinen scannen *)
			X.ADD_COM := 01; (* ADD INFO *)
			X.STORE_TYPE := BYTE#1; (* RRD-Modus *)
		END_IF;

		ERROR_T := BYTE#0;
		ERROR_C := DWORD#0;

		X.UCB.D_MODE := 99;
		UCB(DATA:=X.UCB); (* Buffer löschen *)

		URL_DATA:=STRING_TO_URL(STR:=URL,DEFAULT_PROTOCOL:='',DEFAULT_PATH:=''); (* URL zerlegen *)
		sep_char := OSCAT_BASIC.CHR_TO_STRING(SEP);
		url_query := URL_DATA.QUERY;
		step_1 := 30;

		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_RRD1 - Start: DATE [~2]';
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

	END_IF;

30:	(* LOG-Trigger verarbeiten *)
	trig_tmp := (TRIG_M AND NOT trig_m_last) OR trig_auto OR X.ADD_DATA_REQ;

	IF ENABLE = FALSE THEN (* ENABLE = 0 -> Log Stop *)
		X.ADD_COM := 00;
		step_1 := 00;

		(* ---------------- Debug-Message -----------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'FILE_RRD1 - Stop: DATE [~2]';
			LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------*)

	ELSIF ENABLE THEN
		IF trig_tmp OR trig_stored THEN (* manueller, automatischer oder externer Trigger *)
			trig_stored := FALSE;
			X.ADD_COM := 03; (* ADD DATA *)
			(* ---------------- Debug-Message -----------------------*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'FILE_RRD1 - Trig: DATE [~2]';
				LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
				LOG_MSG();
			END_IF;*)
			(* -----------------------------------------------------------------*)
		ELSE
			X.ADD_COM := 04;	(* ADD DATA REQ *)
		END_IF;
	END_IF;

END_CASE;
X.ADD_DATA_REQ := FALSE; (* externe Anforderung rücksetzen *)

(* ##################################### *)

CASE step_2 OF

00:	IF X.UCB.BUF_DATA_LOST > UDINT#0 THEN
		ERROR_C := DWORD#2; (* Fehler: Bufferüberlauf - Datenmenge reduzieren *)
		ERROR_T := BYTE#6; 	(* Type: Buffer *)
		step_2 := 100;
	ELSIF X.UCB.BUF_USED_MAX > USINT#90 THEN
		ERROR_C := DWORD#1; (* Warnung: Buffer auslastung > 90% *)
		ERROR_T := BYTE#6;	(* Type: Buffer *)
	END_IF;

	IF X.UCB.BUF_COUNT >= USINT_TO_INT(X.ID_MAX) THEN (* genug Einträge vorhanden *)
		URL_DATA.QUERY := url_query; (* original Query wiederherstellen *)
		n := 1;
		WHILE n <= USINT_TO_INT(X.ID_MAX) DO
			X.UCB.D_MODE := 12; (* Element lesen und löschen *)
			UCB(DATA:=X.UCB);
			IF n < USINT_TO_INT(X.ID_MAX) THEN (* Trennzeichen hinzufügen *)
				X.UCB.D_STRING := CONCAT(X.UCB.D_STRING,sep_char);
			END_IF;
			IF LEN(X.UCB.D_STRING) + LEN(URL_DATA.QUERY) <= OSCAT_BASIC.STRING_LENGTH THEN
				URL_DATA.QUERY := CONCAT(URL_DATA.QUERY,X.UCB.D_STRING); (* Query mit LOG-Daten ergänzen *)
				step_2 := 40;
			ELSE
				ERROR_T := BYTE#4; (* LOG-Daten können nicht in URL gepackt werden *)
				ERROR_C := DWORD#1;
				step_2 := 100;
			END_IF;
			n := n + 1;
		END_WHILE;;
	END_IF;

40:	IF DNS_CLIENT.DONE THEN
		step_2 := 60;
	ELSIF (DNS_CLIENT.ERROR > DWORD#0) THEN
		ERROR_C := DNS_CLIENT.ERROR; (* Fehlerbehandlung *)
		ERROR_T := BYTE#1;
		step_2 := 100;
	END_IF;

60:	IF HTTP_GET.DONE THEN
		n := UINT_TO_INT(HTTP_GET.BODY_START);
		IF R_BUF.BUFFER[n] <> BYTE#48 THEN (* ASCII '0' *)
			ERROR_C := DWORD#1;
			ERROR_T := BYTE#3;
		END_IF;
		step_2 := 100;

	ELSIF HTTP_GET.ERROR > DWORD#0 THEN (* Fehlerbehandlung *)
		ERROR_C := HTTP_GET.ERROR;
		ERROR_T := BYTE#2;
		step_2 := 100;
	END_IF;

100:	IF (NOT HTTP_GET.DONE) THEN (* UNLOCK HTTP DATA *)
		step_2 := SEL(ERROR_T = BYTE#0, 120,0);
	END_IF;

120:	IF NOT ENABLE THEN
		step_2 := 0;
	END_IF;

END_CASE;

HTTP_GET(IP_C:=IP_C,S_BUF:=S_BUF,R_BUF:=R_BUF,IP4:=DNS_CLIENT.IP4,GET:=step_2=60,MODE:=2,UNLOCK_BUF:=step_2=100,URL_DATA:=URL_DATA); (* HTTP_GET *)
DNS_CLIENT(IP_C:=IP_C, S_BUF:=S_BUF, R_BUF:=R_BUF, DOMAIN:=URL_DATA.DOMAIN, IP4_DNS:= DNS_IP4, ACTIVATE:=step_2=40); (* DNS-Client *)
IPC(IP:= 0, PORT:= 0 , TIME_OUT:= TIMEOUT, IP_C:= IP_C, S_BUF:= S_BUF, R_BUF:= R_BUF); (* IP_CONTROL *)

trig_m_last := TRIG_M;

(* revision history
ks	01. jan. 2011	rev 1.0
	original version

ks	13. may. 2011	rev 1.1
	ADD_DATA_REQ hinzugefügt

ks	26. july. 2011	rev 1.2
	code optimierung D_MODE 12 und bessere fehlerauswertung

ks	16. apr. 2012	rev 1.3
	neuer überarbeiteter Code

ks	21. mar. 2016	rev 1.4
	nur einmal nach satellitenbausteinen scannen und fehlermeldungen hinzugefügt

*)
END_FUNCTION_BLOCK
