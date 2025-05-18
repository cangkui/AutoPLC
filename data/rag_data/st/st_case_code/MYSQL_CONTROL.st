FUNCTION_BLOCK MYSQL_CONTROL
VAR_IN_OUT
    COM : MYSQL_COM;
    INFO : MYSQL_INFO;
END_VAR

VAR
    IP_C : IP_C;
    URL_DATA : url;
    DNS_CLIENT : DNS_CLIENT;
    IPC : IP_CONTROL;
    step : INT;
    snd_step : INT;
    TON_WAIT : TON;
    last_step : INT;
    idx1 : INT;
    idx2 : INT;
    idx3 : INT;
    dw_tmp : DWORD;
    timeout1 : TIME;
    timeout2 : TIME;
    ip_c_red_disable : BOOL;
    auth_password : STRING[64];
    auth_run : BOOL;
    MYSQL_AUTH : MYSQL_AUTH;
    scramble1_buffer :ARRAY[0..19] OF BYTE;
    scramble2_buffer :ARRAY[0..19] OF BYTE;
    charset_number : BYTE := 16#8;
    max_packet_size : DWORD := 1024;
    client_flags : DWORD := 16#38601;
END_VAR
CASE step OF

00:	IF COM.SQL_CON THEN
		timeout1 := MAX(T#10s,COM.TIMEOUT);
		timeout2 := timeout1 + T#1s;
		INFO.SQL_ERROR := '';
		COM.ERROR_C := DWORD#0;
		COM.ERROR_T := BYTE#0;
		URL_DATA := STRING_TO_URL(STR:=COM.SQL_URL,DEFAULT_PROTOCOL:='',DEFAULT_PATH:='/');
		step := 10;
	END_IF;

10:	IF DNS_CLIENT.DONE THEN
		step := 20;
	ELSIF DNS_CLIENT.ERROR <> DWORD#0 THEN
		COM.ERROR_C := DNS_CLIENT.ERROR;
		COM.ERROR_T := BYTE#1;
		step := 980;
	END_IF;

20:	(* MySQL Steuerkanal einrichten *)
	IF URL_DATA.PORT = WORD#0 THEN
		URL_DATA.PORT := WORD#3306; (* Standard Port *)
	END_IF;
	IP_C.C_PORT := URL_DATA.PORT; (* Portnummer eintragen *)
  	IP_C.C_IP := DNS_CLIENT.IP4; (* IP eintragen *)
  	IP_C.C_MODE := BYTE#0; (* Mode: TCP+ACTIV+PORT *)
  	IP_C.TIME_RESET := TRUE; (* Zeitueberwachung rücksetzen *)
  	IP_C.C_ENABLE := TRUE; (* Verbindungsaufbau freigeben *)
  	IP_C.R_OBSERVE := TRUE; (* Empfangsueberwachung *)
	COM.R_BUF.SIZE := UINT#0;
	COM.SQL_RCV_STATE := BYTE#0;
	step := 30;

30:	IF COM.SQL_RCV_STATE = BYTE#1 THEN
		INFO.SERVER_PROTOCOL_VERSION := COM.R_BUF.BUFFER[4];

		(* ende markierung von string server_version suchen *)
		idx2 := UINT_TO_INT(COM.R_BUF.SIZE) - 1;
		FOR idx1 := 5 TO idx2 DO
			IF COM.R_BUF.BUFFER[idx1] = BYTE#0 THEN EXIT; END_IF;
		END_FOR;
		idx1 := idx1 + 5; (* terminierung und thread-id überspringen *)

		FOR idx2:=0 TO 7 DO	(* Byte 0-7 *)
			scramble1_buffer[idx2] := COM.R_BUF.BUFFER[idx1];
			idx1 := idx1 + 1;
		END_FOR;
		idx1 := idx1 + 1;
		idx2 := idx1 + 1;
		INFO.SERVER_CAPABILITIES := OSCAT_BASIC.WORD_OF_BYTE(COM.R_BUF.BUFFER[idx2], COM.R_BUF.BUFFER[idx1]);
		idx1 := idx1 + 2;
		INFO.SERVER_LANGUAGE := COM.R_BUF.BUFFER[idx1];
		idx1 := idx1 + 1;
		idx2 := idx1 + 1;
		INFO.SERVER_STATUS := OSCAT_BASIC.WORD_OF_BYTE(COM.R_BUF.BUFFER[idx2], COM.R_BUF.BUFFER[idx1]);
		idx1 := idx1 + 15;

		FOR idx2:=8 TO 19 DO	(* Byte 8-19 *)
			scramble1_buffer[idx2] := COM.R_BUF.BUFFER[idx1];
			idx1 := idx1 + 1;
		END_FOR;

		IF INFO.SERVER_PROTOCOL_VERSION < BYTE#10 THEN  (* Error: protocol version zu niedrig *)
			COM.ERROR_T := BYTE#2;
			COM.ERROR_C := DWORD#1;
			step := 980;
		ELSIF URL_DATA.PASSWORD = '' THEN
			step := 50;
		ELSE
			auth_password := URL_DATA.PASSWORD;
			auth_run := TRUE;
			step := 40;
		END_IF;
	END_IF;

40:	MYSQL_AUTH(RUN:= auth_run,PASSWORD:= auth_password ,MESSAGE:= scramble1_buffer,SCRAMBLE:= scramble2_buffer);
	IF NOT auth_run THEN
		step := 50;
	END_IF;

50: (* Client Authentication Packet *)
	(* Byte 0-3 werden später eingetragen *)
	COM.S_BUF.BUFFER[04] := OSCAT_BASIC.BYTE_OF_DWORD(client_flags, BYTE#0); (* client_flags *)
	COM.S_BUF.BUFFER[05] := OSCAT_BASIC.BYTE_OF_DWORD(client_flags, BYTE#1);
	COM.S_BUF.BUFFER[06] := OSCAT_BASIC.BYTE_OF_DWORD(client_flags, BYTE#2);
	COM.S_BUF.BUFFER[07] := OSCAT_BASIC.BYTE_OF_DWORD(client_flags, BYTE#3);
	COM.S_BUF.BUFFER[08] := OSCAT_BASIC.BYTE_OF_DWORD(max_packet_size, BYTE#0); (* max_packet_size *)
	COM.S_BUF.BUFFER[09] := OSCAT_BASIC.BYTE_OF_DWORD(max_packet_size, BYTE#1);
	COM.S_BUF.BUFFER[10] := OSCAT_BASIC.BYTE_OF_DWORD(max_packet_size, BYTE#2);
	COM.S_BUF.BUFFER[11] := OSCAT_BASIC.BYTE_OF_DWORD(max_packet_size, BYTE#3);
	COM.S_BUF.BUFFER[12] := charset_number; (* charset_number *)
	FOR idx1 := 13 TO 35 DO (* 23 bytes filler always 0x00.. *)
		COM.S_BUF.BUFFER[idx1] := BYTE#0;
	END_FOR;
	idx3 := LEN(URL_DATA.USER);
	FOR idx2 :=1 TO idx3 DO
		COM.S_BUF.BUFFER[idx1] := OSCAT_BASIC.CODE(URL_DATA.USER, idx2);
		idx1 := idx1 + 1;
	END_FOR;
	COM.S_BUF.BUFFER[idx1] := BYTE#0; (* String terminierung eintragen *)
	idx1 := idx1 + 1;
	IF LEN(URL_DATA.PASSWORD) = 0 THEN
		COM.S_BUF.BUFFER[idx1] := BYTE#0; (* scramble_buff length *)
	ELSE
		COM.S_BUF.BUFFER[idx1] := BYTE#20; (* scramble_buff length *)
		FOR idx2 :=0 TO 19 DO
			idx1 := idx1 + 1;
			COM.S_BUF.BUFFER[idx1] := scramble2_buffer[idx2];
		END_FOR;
	END_IF;
	COM.S_BUF.SIZE  := INT_TO_UINT(idx1 + 1); (* Sendelänge eintragen *)

	step := 60;

60:	IF COM.SQL_RCV_STATE = BYTE#1 THEN
		INFO.SQL_CONNECTED := TRUE;
		step := 300;
	END_IF;

(* ================================================== *)

300:	step := SEL(COM.SQL_CON,700,310); (* Schritt-Selektion *)

310:	step := 300; (* dummy step *)

(* ================================================== *)

700: (* COM_QUIT *)
	COM.S_BUF.BUFFER[04] := BYTE#1; (* 0x01 COM_QUIT mysql_close *)
	COM.S_BUF.SIZE := UINT#5; (* Sendelänge eintragen *)
	COM.SQL_PACKET_NO := BYTE#255;
	ip_c_red_disable := TRUE;
	step := 710;

710:	IF IP_C.C_STATE = BYTE#0 THEN (* Verbindung durch SQL-Server beendet ? *)
		step := 980;
	END_IF;

(* ================================================== *)

980:IP_C.C_ENABLE := FALSE; (* MYSQL Steuerkanal schliessen *)
	ip_c_red_disable := FALSE;
	INFO.SQL_CONNECTED := FALSE;
	COM.SQL_CON := FALSE;
	COM.S_BUF.SIZE := UINT#0;
	COM.R_BUF.SIZE := UINT#0;
	snd_step := 0;
	step := 990;
	(*####################  DEBUG-MESSAGE  ###################################*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'QUIT: ERROR CODE >~1< TYPE >~2<';
		LOG_CL.PRINTF[1] := DWORD_TO_STRH(COM.ERROR_C);
		LOG_CL.PRINTF[2] := BYTE_TO_STRING(COM.ERROR_T);
		LOG_MSG();
	END_IF;*)
	(*########################################################################*)

990:IF IP_C.C_STATE = BYTE#0 THEN
		step := 0;
	END_IF;

END_CASE;

(* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ *)

IF step >= 30 THEN
	IF COM.R_BUF.SIZE >= UINT#3 AND COM.SQL_RCV_STATE = BYTE#0 THEN (* Daten wurden empfangen ? *)
		IF DWORD_TO_UINT(OSCAT_BASIC.DWORD_OF_BYTE(BYTE#0, COM.R_BUF.BUFFER[2], COM.R_BUF.BUFFER[1], COM.R_BUF.BUFFER[0])) + UINT#4 = COM.R_BUF.SIZE THEN
			COM.SQL_PACKET_NO := COM.R_BUF.BUFFER[3];
			IF COM.R_BUF.BUFFER[4] = BYTE#16#FF THEN
				idx2 := 13; (* Error String - Erstes Zeichen *)
				idx3 := UINT_TO_INT(COM.R_BUF.SIZE) - 1; (* Error String - Letztes Zeichen *)
				FOR idx1 := idx2 TO idx3 DO (* alle ' durch ´ ersetzen *)
					IF COM.R_BUF.BUFFER[idx1] = BYTE#39 THEN
						COM.R_BUF.BUFFER[idx1] := BYTE#96;
					END_IF;
				END_FOR;
				INFO.SQL_ERROR := OSCAT_BASIC.BUFFER_TO_STRING(PT:=ADR(COM.R_BUF.BUFFER),SIZE:=COM.R_BUF.SIZE,START:=INT_TO_UINT(idx2),STOP:=INT_TO_UINT(idx3));
				COM.ERROR_T := BYTE#3;
				COM.ERROR_C := OSCAT_BASIC.DWORD_OF_BYTE(BYTE#0,BYTE#0,COM.R_BUF.BUFFER[6], COM.R_BUF.BUFFER[5]);
				COM.SQL_RCV_STATE := BYTE#2;
				COM.R_BUF.SIZE := UINT#0; (* Empfangsbuffer nach Fehler rücksetzen *)
				step := 980;
			ELSE
				COM.SQL_RCV_STATE := BYTE#1;
				IP_C.R_OBSERVE := FALSE;

				(*####################  DEBUG-MESSAGE  ###################################*)
				(*IF _debug_enable THEN
					LOG_CL.NEW_MSG := 'RCV: size >~1<';
					LOG_CL.PRINTF[1] := UINT_TO_STRING(COM.R_BUF.SIZE);
					LOG_MSG();
				END_IF;*)
				(*########################################################################*)
			END_IF;
		END_IF;
	END_IF;
END_IF;

(* automatisches Datensenden *)

CASE snd_step OF

00:	IF step >= 20 AND COM.S_BUF.SIZE > UINT#0 THEN
		dw_tmp := UINT_TO_DWORD(COM.S_BUF.SIZE - UINT#4); (* packet_length *)
		COM.S_BUF.BUFFER[0] := OSCAT_BASIC.BYTE_OF_DWORD(dw_tmp, BYTE#0);
		COM.S_BUF.BUFFER[1] := OSCAT_BASIC.BYTE_OF_DWORD(dw_tmp, BYTE#1);
		COM.S_BUF.BUFFER[2] := 0;
		COM.SQL_PACKET_NO := COM.SQL_PACKET_NO + 1;
		COM.S_BUF.BUFFER[3] := COM.SQL_PACKET_NO;

		IP_C.R_OBSERVE := TRUE;
		COM.R_BUF.SIZE := UINT#0; (* Empfangsbuffer rücksetzen *)
		COM.SQL_RCV_STATE := BYTE#0;

	        (*####################  DEBUG-MESSAGE  ###################################*)
	        (*IF _debug_enable THEN
	          LOG_CL.NEW_MSG := 'SND: size >~1<';
	          LOG_CL.PRINTF[1] := UINT_TO_STRING(COM.S_BUF.SIZE);
	          LOG_MSG();
	        END_IF;*)
	        (*########################################################################*)

		snd_step := 10;
	END_IF;

10:	IF COM.S_BUF.SIZE = UINT#0 THEN (* warten bis alle daten gesendet sind *)
		snd_step := 0;
	END_IF;

END_CASE;

(* Globale Fehlerbehandlung für Haupt-Routine *)
IF COM.ERROR_T = BYTE#0 AND step > 20 THEN (* noch kein Fehler gespeichert und Hauptroutine aktiv *)
	IF IP_C.ERROR > DWORD#0 AND IP_C.C_ENABLE AND NOT IP_C.TIME_RESET THEN  (* IP Control *)
		IF ip_c_red_disable = FALSE OR IP_C.ERROR <> DWORD#16#FD000000 THEN
			COM.ERROR_C := IP_C.ERROR;
			COM.ERROR_T := BYTE#2;
			step := 980;
		END_IF;
	END_IF;
END_IF;

IF TON_WAIT.Q THEN (* Ablaufstörung - Schrittkette *)
	COM.ERROR_C := INT_TO_DWORD(step); (* aktuellen Schritt eintragen *)
	COM.ERROR_T := BYTE#5;
	step := 980;
END_IF;

(* ######################################################## *)

DNS_CLIENT(IP_C:=IP_C, S_BUF:=COM.S_BUF, R_BUF:=COM.R_BUF, DOMAIN:=URL_DATA.DOMAIN, IP4_DNS:= COM.DNS_IP4, ACTIVATE:=step=10); (* DNS-Client *)
IPC(IP:= DWORD#0, PORT:= WORD#0 , TIME_OUT:= timeout1, IP_C:= IP_C, S_BUF:= COM.S_BUF, R_BUF:= COM.R_BUF); (* MYSQL Steuerkanal *)

TON_WAIT(IN:=step = last_step AND step > 0,PT:=timeout2);
last_step := step;

(* revision history
ks	20. apr. 2012	rev 1.0
	original version
ks	14. oct. 2013	rev 1.1
	DNS_CLIENT Fehler durch zugriffskonflikt zwischen dns_client und mySQL auf S_BUF

*)
END_FUNCTION_BLOCK
