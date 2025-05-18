FUNCTION_BLOCK DLOG_STORE_MYSQL
VAR_IN_OUT
    SQL_INFO : MYSQL_INFO;
    X : DLOG_DATA;
END_VAR

VAR_INPUT
    ENABLE : BOOL;
    TRIG_M : BOOL;
    TRIG_T : TIME;
    DTI : DT;
    URL : STRING(OSCAT_BASIC.STRING_LENGTH);
    DB_NAME : STRING(64);
    TB_NAME : STRING(64);
    AUTO_DELETE : UDINT;
    TIMEOUT : TIME;
    DNS_IP4 : DWORD;
END_VAR

VAR_OUTPUT
    ERROR_C : DWORD;
    ERROR_T : BYTE;
END_VAR

VAR
    UCB : UNI_CIRCULAR_BUFFER;
    MYSQL_CONTROL : MYSQL_CONTROL;
    trig_auto : BOOL;
    trig_tmp : BOOL;
    trig_m_last : BOOL;
    dti_last : DT := DATE_AND_TIME#2070-2-6-6:28:15;
    enable_last : BOOL;
    step : INT;
    idx : INT;
    n : USINT;
    snd_text : STRING(OSCAT_BASIC.STRING_LENGTH);
    snd_size : INT;
    sql_command : BYTE;
    next_step : INT;
    i1 : INT;
    i2 : INT;
    y : MYSQL_COM;
    del_dt : UDINT;
    del_start : BOOL;
    run : BOOL;
END_VAR
X.DTI := DTI; (* Datum / Zeit eintragen *)
X.CLOCK_TRIG := DTI <> dti_last;
dti_last := DTI;
trig_auto := FALSE;
IF X.CLOCK_TRIG THEN (* Impuls in jeder Sekunde *)
	del_start := DT_TO_UDINT(DTI) - del_dt > AUTO_DELETE AND AUTO_DELETE > UDINT#0;
	IF TRIG_T >= T#1000ms THEN (* automatischer Trigger *)
		trig_auto := DT_TO_UDINT(DTI) MOD (TIME_TO_UDINT(TRIG_T) / UDINT#1000) = UDINT#0;
	END_IF;
END_IF;
X.ADD_COM := 00;

CASE step OF

00:	IF ENABLE AND enable_last = FALSE THEN
		y.SQL_URL := URL;
		y.TIMEOUT := TIMEOUT;
		y.DNS_IP4 := DNS_IP4;
		y.SQL_CON := TRUE;

		X.UCB.D_MODE := 99; (* Buffer reset - Alles löschen *)
		UCB(DATA:=X.UCB);

		IF X.ID_MAX = USINT#0 THEN (* nur einmal nach Satellitenbausteinen scannen *)
			X.ADD_COM := 01; (* ADD INFO *)
			X.STORE_TYPE := BYTE#5; (* MySQL-Modus *)
		END_IF;

		step := 05;
	END_IF;

05:	X.ADD_COM := 02; (* ADD HEADER *)
	step := 10;

10:	IF SQL_INFO.SQL_CONNECTED THEN
		sql_command := BYTE#3; (* CREATE DATABASE IF NOT EXISTS `db_name` *)
		snd_text :=  CONCAT('CREATE DATABASE IF NOT EXISTS `' ,DB_NAME);
		snd_text :=  CONCAT(snd_text, '`');
		next_step := 20;
	END_IF;

20:	sql_command := BYTE#2; (* 0x02 COM_INIT_DB mysql_select_db *)
	snd_text := DB_NAME;
	next_step := 30;

30:	(* CREATE TABLE IF NOT EXISTS tb_name (Timestamp TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP, n1,n2) *)
	sql_command := BYTE#0;
	idx := 5;
	n := USINT#0;
	WHILE sql_command = BYTE#0 DO
		IF n = USINT#0 THEN
			snd_text := CONCAT('CREATE TABLE IF NOT EXISTS `', TB_NAME);
			snd_text := CONCAT(snd_text,'` (`ID` BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY ,');
			snd_text := CONCAT(snd_text,' store_timestamp TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP');
		ELSIF n <= X.ID_MAX THEN
			X.UCB.D_MODE := 12; (* Element lesen und löschen *)
			UCB(DATA:=X.UCB);
			snd_text := CONCAT(', `',X.UCB.D_STRING);
			CASE WORD_TO_INT(SHR(X.UCB.D_HEAD,8)) OF
			02:	snd_text := CONCAT(snd_text, '` FLOAT'); (* REAL *)
			06:	snd_text := CONCAT(snd_text, '` BOOL'); (* BOOL *)
			08:	snd_text := CONCAT(snd_text, '` INT'); (* DINT *)
			13:	snd_text := CONCAT(snd_text, '` DATETIME'); (* DT *)
			ELSE
				snd_text := CONCAT(snd_text, '` VARCHAR(80)'); (* STRING(80) *)
			END_CASE;
		ELSE
			snd_text := ')';
			sql_command := BYTE#3;
			snd_size := idx + LEN(snd_text);
			next_step := 40;
		END_IF;
		OSCAT_BASIC._STRING_TO_BUFFER(str:=snd_text,pos:=idx,pt:=ADR(y.S_BUF.BUFFER),size:=SIZEOF(y.S_BUF.BUFFER)); (* String in puffer kopieren *)
		idx := idx + LEN(snd_text);
		n := n + USINT#1;

		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'dlog_mysql: create >~1<';
			LOG_CL.PRINTF[1] := snd_text;
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)
	END_WHILE;

40:	IF del_start THEN
		del_start := FALSE;
		sql_command := BYTE#3; (* DELETE FROM `tb_name` WHERE TIMESTAMPDIFF(SECOND, Timestamp, NOW()) > xxx *)
		snd_text := CONCAT('DELETE FROM `', TB_NAME);
		snd_text := CONCAT(snd_text, '` WHERE TIMESTAMPDIFF(SECOND, store_timestamp, NOW()) > ');
		snd_text := CONCAT(snd_text, UDINT_TO_STRING(AUTO_DELETE));
		next_step := 40;
		del_dt := DT_TO_UDINT(DTI);
		(*####################  DEBUG-MESSAGE  ###################################*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'dlog_mysql: # delete time:~1 >~2<';
			LOG_CL.PRINTF[1] := UDINT_TO_STRING(AUTO_DELETE);
			LOG_CL.PRINTF[2] := snd_text;
			LOG_MSG();
		END_IF;*)
		(*########################################################################*)

         (* INSERT INTO `tb_name` VALUES (CURRENT_TIMESTAMP,  "n1","n2") *)
	ELSIF X.UCB.BUF_COUNT >= USINT_TO_INT(X.ID_MAX) AND step > 0 THEN (* Neue Datenreihe(n) vorhanden *)
		idx := 5;
		n := USINT#0;
		sql_command := BYTE#0;
		WHILE sql_command = BYTE#0 DO
			IF n = USINT#0 THEN
				snd_text := CONCAT('INSERT INTO `', TB_NAME);
				snd_text := CONCAT(snd_text,'` VALUES (NULL, CURRENT_TIMESTAMP');
			ELSIF n <= X.ID_MAX THEN
				X.UCB.D_MODE := 12; (* Element lesen und löschen *)
				UCB(DATA:=X.UCB);
				snd_text := CONCAT(', "',X.UCB.D_STRING);
				snd_text := CONCAT(snd_text, '"');
			ELSE
				snd_text := ')';
				sql_command := BYTE#3;
				snd_size := idx + LEN(snd_text);
				next_step := 40;
			END_IF;
			OSCAT_BASIC._STRING_TO_BUFFER(str:=snd_text,pos:=idx,pt:=ADR(y.S_BUF.BUFFER),size:=SIZEOF(y.S_BUF.BUFFER)); (* String in puffer kopieren *)
			idx := idx + LEN(snd_text);
			n := n + USINT#1;

			(*####################  DEBUG-MESSAGE  ###################################*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'dlog_mysql: insert >~1<';
				LOG_CL.PRINTF[1] := snd_text;
				LOG_MSG();
			END_IF;*)
			(*########################################################################*)
		END_WHILE;
	END_IF;

	IF ENABLE THEN 	(* LOG-Trigger verarbeiten *)
		trig_tmp := (TRIG_M AND NOT trig_m_last) OR trig_auto OR X.ADD_DATA_REQ;

		IF trig_tmp THEN (* manueller, automatischer oder externer Trigger *)
			X.ADD_COM := 03; (* ADD DATA *)
			(* ---------------- Debug-Message -----------------------*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'dlog_mysql: Trigger: DATE [~2]';
				LOG_CL.PRINTF[2] := DT_TO_STRING(DTI);
				LOG_MSG();
			END_IF;*)
			(* -----------------------------------------------------------------*)
		ELSE
			X.ADD_COM := 04;	(* ADD DATA REQ *)
		END_IF;
	ELSE
		y.SQL_CON := FALSE; (* MySQL Verbindung beenden *)
		step := 0;
	END_IF;
	X.ADD_DATA_REQ := FALSE;

(* ~~~~~~~~~~~~~~~~~~~~~~ *)

199:IF y.SQL_RCV_STATE = BYTE#1 THEN (* OK *)
		step := next_step;
	ELSIF y.SQL_RCV_STATE = BYTE#2 THEN  (* FEHLER *)
		step := 0;
	END_IF;

END_CASE;

(* ########################## *)
(* SQL-Kommando ausführen *)
IF sql_command > BYTE#0 THEN

	IF snd_size > 0 THEN
		y.S_BUF.SIZE := INT_TO_UINT(snd_size);
		snd_size := 0;
	ELSE
		OSCAT_BASIC._STRING_TO_BUFFER(str:=snd_text,pos:=5,pt:=ADR(y.S_BUF.BUFFER),size:=SIZEOF(y.S_BUF.BUFFER)); (* String in puffer kopieren *)
		y.S_BUF.SIZE := INT_TO_UINT(5 + LEN(snd_text));
		snd_text := '';
	END_IF;

	IF y.S_BUF.SIZE > SIZEOF(y.S_BUF.BUFFER) THEN (* buffer überlauf erkannt -> Abbruch *)
		y.S_BUF.SIZE := 0;
		sql_command := BYTE#0;
		y.SQL_CON := FALSE; (* MySQL Verbindung beenden *)
		step := 999; (* Dummy-Schritt *)
	ELSE

		i2 := UINT_TO_INT(y.S_BUF.SIZE) - 1; (* alle " zeichen mit ' ersetzen *)
		FOR i1 := 5 TO i2 DO
			IF y.S_BUF.BUFFER[i1] = BYTE#34 THEN
				y.S_BUF.BUFFER[i1] := BYTE#39;
			END_IF;
		END_FOR;
	
		y.S_BUF.BUFFER[4] := sql_command;
		y.SQL_PACKET_NO := BYTE#255;
		sql_command := BYTE#0;
		step := 199;
	END_IF;
END_IF;

(* ########################## *)

MYSQL_CONTROL(COM:= y, INFO:=SQL_INFO);

(* Fehlerauswertung *)
IF step = 999 THEN	(* Buffer überlauf *)
	ERROR_C := DWORD#3; (* Netzwerkbuffer überlauf *)
	ERROR_T := BYTE#6; 	(* Type: Buffer *)
ELSIF y.ERROR_T > BYTE#0 THEN	(* SQL Error *)
	ERROR_C := y.ERROR_C;
	ERROR_T := y.ERROR_T;
	step := 0;
ELSIF X.UCB.BUF_DATA_LOST > UDINT#0 THEN
	ERROR_C := DWORD#2; (* Fehler: Bufferüberlauf - Datenmenge reduzieren *)
	ERROR_T := BYTE#6; 	(* Type: Buffer *)
ELSIF X.UCB.BUF_USED_MAX > USINT#90 THEN
	ERROR_C := DWORD#1; (* Warnung: Buffer auslastung > 90% *)
	ERROR_T := BYTE#6;	(* Type: Buffer *)
ELSE
	ERROR_C := DWORD#0;
	ERROR_T := BYTE#0;
END_IF;



(* ########################## *)

trig_m_last := TRIG_M;
enable_last := ENABLE;
run := step > 0;

(* revision history
ks	20. apr. 2012	rev 1.0
	original version

ks	17. jan. 2014	rev 1.1
	DELTA Error 

ks	8. mar. 2016	rev 1.2
	nur einmal nach satellitenbausteinen scannen

ks	21. mar. 2016	rev 1.3
	prüfung auf bufferüberlauf und fehlermeldungen hinzugefügt

*)
END_FUNCTION_BLOCK
