FUNCTION_BLOCK IP_CONTROL
VAR_IN_OUT
    IP_C: IP_C;                         // IP_Control Verwaltungsstruktur
    S_BUF: NETWORK_BUFFER;               // Externer Sendbuffer
    R_BUF: NETWORK_BUFFER;               // Externer Receivebuffer
END_VAR

VAR_INPUT
    IP: DWORD;                          // IP Adresse
    PORT: WORD;                         // Port Nummer
    TIME_OUT: TIME;                     // Ueberwachungszeit Connect / Send / Receive
END_VAR

VAR
    c_time: UDINT;                      // CONNECT-Time (Zeit bei nicht vorhandener Verbindung)
    s_time: UDINT;                      // SEND-Time (Zeit seit letzten Datensenden)
    r_time: UDINT;                      // RECEIVE-Time (Zeit seit letztem Datenempfang)
    c_enable: BOOL;                     // Connect: freigabe
    c_ip: DWORD;                        // Connect: aktuelle IP-Adresse
    c_port: WORD;                       // Connect: aktuelle Port-Nummer
    c_mode: BYTE;                       // Connect: aktueller Mode UDP/TCP......
    c_status: BYTE;                     // Connect: Status (Errorcode)
    c_ready: BOOL;                      // Connect: Verbindung aufgebaut
    c_ready_old: BOOL;                  // Connect: letzter Zustand von c_ready für Flankenauswertung
    c_select: BOOL;                     // non blocking mode - abfrage über select durchführen
    c_fdWrite: SOCKET_FD_SET;           // File descriptor set for writing
    c_Timeout: SOCKET_TIMEVAL;          // Timeout for socket operations
    s_total: INT;                       // Send: Gesamtanzahl Bytes zum senden
    s_cur_pos: INT;                     // Send: aktuelle Sendeposition
    s_cur_size: INT;                    // Send: aktuelle Sendegroesse
    s_max_size: INT;                    // Send: maximale Telegrammlänge fuer Senden
    s_start: BOOL;                      // Send: Sende modus aktivieren
    s_active: BOOL;                     // Send: Daten senden ist aktiv
    s_req: BOOL;                        // Send: Start Daten senden
    s_done: BOOL;                       // Send: ohne Fehler beendet
    s_status: BYTE;                     // Send: Status (Errorcodes)
    r_status: BYTE;                     // Receive: Status (Errorcodes)
    r_count: INT;                       // Receive: Anzahl empfangener Bytes
    r_offset: INT;                      // Receive: aktuelle Position in externen Receivebuffer
    r_max_size: INT;                    // Receive: groesse des externen Empfangbuffers
    new_connection: BOOL;               // Neue Verbindungsparameter vorhanden
    tx: DWORD;                          // letzter zeitwert
    error_time: DWORD;                  // aktueller Timeout-Wert
    state: BYTE;                        // schrittmerker für connect
    server_socket: DINT := SOCKET_INVALID; // Server socket, initialized to invalid
    socket: DINT := SOCKET_INVALID;     // Socket, initialized to invalid
    sockaddr: SOCKADDRESS;              // Socket address structure
    sockaddr_size: DWORD;               // Size of the socket address structure
    plc_841: BOOL;                      // Generic boolean for plc_841
    plc_881: BOOL;                      // Generic boolean for plc_881
    dint_true: DINT := 1;               // DINT checking variable set to 1
    bytes_received: DINT;                // Bytes received from the socket
    bytes_sent: DINT;                   // Bytes sent to the socket
    udp_mode: BOOL;                     // Indicates if UDP mode is active
END_VAR

VAR CONSTANT
    STOP: BYTE := 0;                    // Constant for STOP state
    TS_INIT: BYTE := 1;                 // Constant for initialization state
    TS_OPEN: BYTE := 2;                 // Constant for open state
    UDP_INIT: BYTE := 31;               // Constant for UDP initialization
    TC_INIT: BYTE := 51;                // Constant for TCP initialization
    TC_CONNECT: BYTE := 52;             // Constant for TCP connection
    C_WAIT: BYTE := 250;                // Constant for wait state
    C_CLOSE: BYTE := 251;               // Constant for close state
END_VAR
(* ---------------------- Debug-Message ----------------------------*)
(*IF _debug_enable THEN
	IF IP_C.FIFO.ID > BYTE#0 THEN
		debug_index := IP_C.FIFO.NR;
		debug_ID := BYTE_TO_INT(IP_C.FIFO.X[debug_index]);
	END_IF;
END_IF;*)
(* -----------------------------------------------------------------------------*)

(* read system time *)
tx := OSCAT_BASIC.T_PLC_MS();

IF IP_C.C_PORT = 0 THEN
	IP_C.C_PORT := PORT; (* uebergebenen PORT durch parametrierten PORT ersetzen *)
END_IF;
IF IP_C.C_IP = 0 AND IP_C.C_MODE < 4 THEN
	IP_C.C_IP := IP; (* uebergebene IP durch paramtrierte IP ersetzen *)
END_IF;

new_connection := c_ip <> IP_C.C_IP OR c_mode <> IP_C.C_MODE OR c_port <> IP_C.C_PORT; (* neue Verbindungs erwünscht ? *)
c_enable := IP_C.C_ENABLE AND NOT new_connection;

IF IP_C.C_ENABLE AND IP_C.ERROR = 0 AND state = STOP THEN (* Freigabe,kein Fehler,kein Socket offen *)
	c_ip   := IP_C.C_IP;
	c_mode := IP_C.C_MODE;
	c_port := IP_C.C_PORT;
	plc_841 := SYSLIBSOCKETS_OPTION.1; (* Bit 1 = WAGO 750-841 *)
	plc_881 := SYSLIBSOCKETS_OPTION.2; (* Bit 1 = WAGO 750-881 *)
	udp_mode := C_MODE.0; (* Mode 1,3,5 = udp *)

	error_time := TIME_TO_DWORD(MAX(T#200ms,TIME_OUT));
	IP_C.TIME_RESET := TRUE; (* Alle Timer ruecksetzen *)

	r_max_size := SIZEOF(R_BUF.BUFFER);
	s_max_size := SIZEOF(S_BUF.BUFFER);

	IF C_MODE.0 THEN (* Mode: 1,3,5 = UDP CLIENT/SERVER *)
		state := UDP_INIT;
	ELSIF C_MODE = 0 THEN (* Mode: 0 = TCP CLIENT *)
		state := TC_INIT;
	ELSE
		state := TS_INIT; (* Mode: 2,4 = TCP SERVER *)
	END_IF;

	(* ---------------------- Debug-Message ----------------------------*)
	(*IF _debug_enable THEN
		LOG_CL.NEW_MSG := 'IP_CON: MODE ~1 / IP ~2 / PORT ~3 /ID ~6';
		LOG_CL.PRINTF[1] := BYTE_TO_STRING(C_MODE);
		LOG_CL.PRINTF[2] := IP4_TO_STRING(c_ip);
		LOG_CL.PRINTF[3] := WORD_TO_STRING(c_port);
		LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
		LOG_MSG();
	END_IF;*)
	(* -----------------------------------------------------------------------------*)

END_IF;

(* ---------------------- Debug-Message ----------------------------*)
(*IF  _debug_enable THEN
	IF state <> state_last THEN
		LOG_CL.NEW_MSG := 'IP_STEP ~1 -IP_ID ~6';
		LOG_CL.PRINTF[1] := BYTE_TO_STRING(state);
		LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
		LOG_MSG();
		state_last := state;
	END_IF;
END_IF;*)
(* -----------------------------------------------------------------------------*)

IF IP_C.TIME_RESET THEN (* Alle Timer ruecksetzen *)
	IP_C.TIME_RESET := FALSE; (* Reset rücksetzen *)
	c_time := tx; (* Connect Timer ruecksetzen *)
	s_time := tx; (* Send Timer ruecksetzen *)
	r_time := tx; (* Receive Timer ruecksetzen *)

	c_status := BYTE#0; (* letzten Connect Fehler rücksetzen *)
	s_status := BYTE#0; (* letzten Send Fehler rücksetzen *)
	r_status := BYTE#0; (* letzten Receive Fehler rücksetzen *)
END_IF;

(* ---------------------------------------------- *)

CASE state OF

UDP_INIT: (* UDP_CLIENT + SERVER *)
	socket := SysSockCreate(SOCKET_AF_INET, SOCKET_DGRAM, SOCKET_IPPROTO_IP);
	IF socket < 0 THEN
		c_status := 1; (* SysSockCreate failed *)
		state := C_CLOSE;
	ELSE
		SysSockSetOption(socket, SOCKET_SOL, SOCKET_SO_BROADCAST, ADR(dint_true), SIZEOF(dint_true)); (* allow broadcast *)
		SysSockIoctl(socket, SOCKET_FIONBIO, ADR(dint_true)); (* put socket in non-blocking mode *)
		sockaddr.sin_family := SOCKET_AF_INET;
		sockaddr.sin_port := SysSockHtons(c_port);
		sockaddr.sin_addr := SEL(c_mode = 1, SOCKET_INADDR_ANY, SysSockNtohl(c_ip));

		IF c_mode >= 2 THEN (* SERVER Mode *)
			IF SysSockBind(socket, ADR(sockaddr), SIZEOF(sockaddr)) THEN
				c_ready := TRUE; (* Connected *)
				state:= C_WAIT;
			ELSE
				c_status := 2; (* SysSockBind failed *)
				state := C_CLOSE;
			END_IF;
		ELSE
			c_ready := TRUE; (* Connected *)
			state:= C_WAIT;
		END_IF;
	END_IF;

(* ---------------------------------------------- *)

TC_INIT: (* TCP-CLIENT (AKTIV) *)
	socket := SysSockCreate(SOCKET_AF_INET, SOCKET_STREAM, SOCKET_IPPROTO_IP);
	IF socket < 0 THEN
		c_status := 1;  (* SysSockCreate failed *)
		state := C_CLOSE;
	ELSE
		IF plc_841 OR plc_881 THEN
			SysSockSetOption(socket,6,SOCKET_TCP_NODELAY, ADR(dint_true), SIZEOF(dint_true)); (* Set Push-Bit *)
		END_IF;
		SysSockIoctl(socket, SOCKET_FIONBIO, ADR(dint_true)); (* NonBlocking mode *)
		sockaddr.sin_family := SOCKET_AF_INET;
		sockaddr.sin_port := SysSockNtohs(c_port);
		sockaddr.sin_addr := SysSockNtohl(c_ip);
		c_select := FALSE;
		state := TC_CONNECT;
	END_IF;

(* ---------------------------------------------- *)

TC_CONNECT:
	IF NOT c_enable OR c_status = 255 THEN (* keine freigabe oder connect timeout *)
			state:= C_CLOSE;
	ELSE
		IF c_select = FALSE THEN
			IF SysSockConnect(socket,ADR(sockaddr),SIZEOF(sockaddr)) THEN
				c_ready := TRUE; (* Connected *)
				state:= C_WAIT;
			ELSE
				(* Connect durchgeführt, aber noch nicht verbunden, dann optional mit select-abfrage weitermachen *)
				c_select := SYSLIBSOCKETS_OPTION.0; (* Bit 0 = TCP-Client - use SysSockselect *)
			END_IF;
		ELSE
			c_Timeout.tv_sec := 0;
			c_Timeout.tv_usec := 0;
			c_fdWrite.fd_count := 1;
			c_fdWrite.fd_array[0] := socket;
			IF SysSockSelect(SOCKET_FD_SETSIZE,0,ADR(c_fdWrite),0, ADR(c_Timeout)) > 0 THEN
				c_ready := TRUE; (* Connected *)
				state:= C_WAIT;
			END_IF;
		END_IF;
	END_IF;

(* ---------------------------------------------- *)

TS_INIT: (* TCP-SERVER (PASSIV) *)
	server_socket := SysSockCreate(SOCKET_AF_INET, SOCKET_STREAM, SOCKET_IPPROTO_IP); (* Create Socket *)
	IF server_socket = SOCKET_INVALID THEN
		c_status := 1; (* SysSockCreate failed *)
		state:= C_CLOSE;
	ELSE (* Address bindings *)
		SysSockSetOption( server_socket, SOCKET_SOL, SOCKET_SO_REUSEADDR, ADR(dint_true), SIZEOF(dint_true) );
		sockaddr.sin_family:= SOCKET_AF_INET;
		sockaddr.sin_port := SysSockHtons(c_port);
		sockaddr.sin_addr := SOCKET_INADDR_ANY;
		IF NOT SysSockBind(server_socket, ADR(sockaddr), SIZEOF(sockaddr)) THEN
			c_status := 2; (* SysSockBind failed *)
			state := C_CLOSE;
		ELSIF NOT SysSockListen(server_socket, 1) THEN (* erzeugt Warteschlange für eine Verbindung *)
			c_status := 3; (* SysSockListen failed *)
			state:= C_CLOSE;
		ELSE
			SysSockIoctl(server_socket, SOCKET_FIONBIO, ADR(dint_true)); (* NonBlocking mode *)
			state:= TS_OPEN;
		END_IF;
	END_IF;

(* ---------------------------------------------- *)

TS_OPEN: (* Waiting for incomming connection *)
	sockaddr_size := SIZEOF(sockaddr);
	socket := SysSockAccept(server_socket, ADR(sockaddr), ADR(sockaddr_size));

	IF socket <> SOCKET_INVALID THEN (* neuer Verbindungswunsch vorhanden *)
		IF c_mode = BYTE#2 AND sockaddr.sin_addr <> SysSockNtohl(c_ip) THEN (* Mode 2 = TCP + PASSIVE + PORT + IP  auf richtige Partner-IP prüfen *)
			state:= C_CLOSE;
		ELSE
			IF plc_841 OR plc_881 THEN
				SysSockSetOption(socket, 6, SOCKET_TCP_NODELAY, ADR(dint_true), SIZEOF(dint_true)); (* Set Push-Bit *)
			END_IF;
			SysSockIoctl(socket, SOCKET_FIONBIO, ADR(dint_true)); (* NonBlocking mode *)
			c_ready := TRUE; (* Connected *)
			state := C_WAIT;
		END_IF;
	ELSIF NOT c_enable THEN (* keine freigabe mehr ? *)
			state:= C_CLOSE;
	END_IF;

(* ---------------------------------------------- *)

C_WAIT: (* Warten auf Anforderung für Beenden *)
	IF NOT c_enable OR NOT c_ready THEN
		state:= C_CLOSE;
	END_IF

(* ---------------------------------------------- *)

C_CLOSE: (* Close the Sockets *)
	c_ready := FALSE;
	IF socket >= 0 THEN
		IF (NOT plc_841) AND (NOT plc_881) THEN
			SysSockShutdown(socket, 0); (* 870 *)
		END_IF;
		SysSockClose(socket);
		socket := SOCKET_INVALID;
	END_IF
	IF server_socket >= 0 THEN
		IF (NOT plc_841) AND (NOT plc_881) THEN
			SysSockShutdown(server_socket, 0); (* 870 *)
		END_IF;
		SysSockClose(server_socket);
		server_socket := SOCKET_INVALID;
	END_IF
	state:= STOP;
END_CASE;

(* ---------------------------------------------- *)

IF c_ready THEN
	IF NOT c_ready_old THEN (* Verbindung aufgebaut - positive Flanke erkannt *)
		IP_C.C_STATE := 254;
		IP_C.TIME_RESET := TRUE; (* Alle Timer ruecksetzen *)
		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'IP_CON: CONNECTED -IP_ID ~6';
			LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------------------*)
	ELSE
		IP_C.C_STATE := 255; (* Verbindung ist aufgebaut *)
	END_IF;
ELSE
	IF c_ready_old THEN (* Verbindung abgebaut - negative Flanke erkannt *)
		IP_C.C_STATE := 1;
		IP_C.MAILBOX[1] := 0; (* Receive Info rücksetzen *)
		IP_C.MAILBOX[2] := 0; (* Send-Pause rücksetzen *)
		IP_C.MAILBOX[3] := 0; (* Receive-Pause rücksetzen *)
		IF NOT new_connection THEN
			s_active := FALSE; (* Sendeaktivitäten stoppen *)
			s_req := FALSE; (* Sendefreigabe rücksetzen *)
			S_BUF.SIZE := 0; (* Sendelänge löschen *)
			(*IP_C.TIME_RESET := TRUE;*) (* Alle Timer ruecksetzen *)
		END_IF;
		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'IP_CON: DISCONNECTED';
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------------------*)
	ELSE
		IP_C.C_STATE := 0; (* Verbindung ist abgebaut *)
	END_IF;
END_IF;

c_ready_old := c_ready;

(* Der Datenempfang wird angehalten solange ein Sendevorgang läuft *)
(* sonst kann es vorkommen das die Antwort-daten schon empfangen werden, *)
(* bevor der Sendevorgang vom programm abgeschlossen wurde *)

IF c_ready AND NOT s_active AND IP_C.MAILBOX[3] = 0 THEN
	IF R_BUF.SIZE >= INT_TO_UINT(r_max_size) THEN
		R_BUF.SIZE := UINT#0;
		r_status := 254; (* Empfangsbuffer wegen Überlauf rückgesetzt *)
	END_IF;

	r_offset := SEL(udp_mode,UINT_TO_INT(R_BUF.SIZE),0);
	r_count := r_max_size - r_offset;

	IF udp_mode THEN
		bytes_received := SysSockRecvFrom(socket, ADR(R_BUF.BUFFER), r_count, 0, ADR(sockaddr), SIZEOF(sockaddr));

		IF c_mode = BYTE#3 AND sockaddr.sin_addr <> SysSockNtohl(c_ip) THEN (* Mode 3 = UDP + PASSIVE + PORT + IP  auf richtige Partner-IP prüfen *)
			bytes_received := 0;
			R_BUF.SIZE := UINT#0;
			sockaddr.sin_port := SysSockNtohs(c_port);
			sockaddr.sin_addr := SysSockNtohl(c_ip);
		END_IF;

	ELSE
		bytes_received := SysSockRecv(socket, ADR(R_BUF.BUFFER[r_offset]), r_count, 0);
	END_IF;

	(*UDP: IPC returns -1 if no data available or error occured,841 returns -1 if error occured and 0 if no data available*)
	CASE bytes_received OF
	-2147483648..-1: ; (* No data available or error occured *)
	 0: IF NOT udp_mode THEN (* TCP-Mode: socket wurde von remote geschlossen *)
			c_status := 253;
			c_ready := FALSE;
		END_IF;
	ELSE (* daten wurden empfangen *)
  		r_time := tx; (* Receive Timer ruecksetzen *)
		R_BUF.SIZE := INT_TO_UINT(r_offset + DINT_TO_UINT(bytes_received)); (* aktuelle buffersize eintragen *)
		IP_C.MAILBOX[1] := IP_C.MAILBOX[1] + 1; (* Receive Info *)
		IF IP_C.MAILBOX[1] = 0 THEN IP_C.MAILBOX[1] := 1; END_IF;

		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'IP_REC DATA: POS: ~1 : SIZE: ~2 TOTAL ~3 -IP_ID ~6';
			LOG_CL.PRINTF[1] := INT_TO_STRING(r_offset);
			LOG_CL.PRINTF[2] := INT_TO_STRING(r_count);
			LOG_CL.PRINTF[3] := UINT_TO_STRING(R_BUF.SIZE);
			LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------------------*)

	END_CASE
END_IF;

(* ---------------------------------------------- *)

IF S_BUF.SIZE > 0 THEN
	IF c_ready AND c_enable THEN
		IF NOT s_active AND IP_C.MAILBOX[2] = 0 THEN
			(* Gesamtanzahl an Bytes limitieren und uebergeben *)
			s_total := LIMIT(0,UINT_TO_INT(S_BUF.SIZE),r_max_size);
			s_cur_pos  := 0;
			s_cur_size := 0;
			s_start := TRUE; (* Daten kopieren und senden anstossen *)
			s_active := TRUE; (* Sende-Betrieb aktivieren *)
			(* ---------------------- Debug-Message ----------------------------*)
			(*IF _debug_enable THEN
				LOG_CL.NEW_MSG := 'IP_SEN INIT: TOTAL: ~1 -IP_ID ~6';
				LOG_CL.PRINTF[1] := INT_TO_STRING(S_TOTAL);
				LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
				LOG_MSG();
			END_IF;*)
			(* -----------------------------------------------------------------------------*)
		END_IF;

		IF s_done OR s_start THEN (* Daten senden aktiv *)

			s_start := FALSE;
			s_cur_pos := s_cur_pos + s_cur_size; (* neue Sende-Datenposition bestimmen *)
			IF s_total > s_cur_pos THEN (* weitere Daten zum senden vorhanden ? *)
				s_cur_size := s_total - s_cur_pos;
				s_cur_size := LIMIT(0, s_cur_size, s_max_size); (* Telegrammlänge begrenzen *)
				(* ---------------------- Debug-Message ----------------------------*)
				(*IF _debug_enable THEN
					LOG_CL.NEW_MSG := 'IP_SEN DATA: POS: ~1 : SIZE: ~2 TOTAL ~3 -IP_ID ~6';
					LOG_CL.PRINTF[1] := INT_TO_STRING(S_CUR_POS);
					LOG_CL.PRINTF[2] := INT_TO_STRING(S_CUR_SIZE);
					LOG_CL.PRINTF[3] := INT_TO_STRING(S_TOTAL);
					LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
					LOG_MSG();
				END_IF;*)
				(* -----------------------------------------------------------------------------*)
				s_time := tx; (* Send Timer ruecksetzen *)
				r_time := tx; (* Receive Timer ruecksetzen *)
				s_req := TRUE;

			ELSE
				(* keine weitere Sendedaten vorhanden -> Senden beenden *)
				s_req := FALSE;
				s_active := FALSE;
				S_BUF.SIZE := 0;
				(* ---------------------- Debug-Message ----------------------------*)
				(*IF _debug_enable THEN
					LOG_CL.NEW_MSG := 'IP_SEN FINISH -IP_ID ~6';
					LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
					LOG_MSG();
				END_IF;*)
				(* -----------------------------------------------------------------------------*)
			END_IF;
		END_IF;
	END_IF;
END_IF;

IF s_req THEN
	s_done := FALSE;
	IF udp_mode THEN
		bytes_sent := SysSockSendTo(socket, ADR(S_BUF.BUFFER[s_cur_pos]), s_cur_size, 0, ADR(sockaddr), SIZEOF(sockaddr));
	ELSE
		bytes_sent := SysSockSend(socket, ADR(S_BUF.BUFFER[s_cur_pos]), s_cur_size, 0);
	END_IF;

	CASE bytes_sent OF
	-2147483648..-1:	; (* ERROR: Noch nicht bereit um Daten aufzunehmen -> nächsten Zyklus wieder probieren *)
	 0: IF NOT udp_mode THEN (* TCP-Mode: socket wurde von remote geschlossen *)
			c_status := 253;
			c_ready := FALSE;
		END_IF;
	ELSE
		s_done := TRUE;
		s_cur_size := DINT_TO_INT(bytes_sent); (* Anzahl der wirklich versendeten Bytes übernehmen *)
		(* ---------------------- Debug-Message ----------------------------*)
		(*IF _debug_enable THEN
			LOG_CL.NEW_MSG := 'IP_SEN SEND_COUNT ~1 -IP_ID ~6';
			LOG_CL.PRINTF[1] := DINT_TO_STRING(s_cur_size);
			LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
			LOG_MSG();
		END_IF;*)
		(* -----------------------------------------------------------------------------*)
	END_CASE
END_IF;

(* ---------------------------------------------- *)

IF IP_C.R_OBSERVE = FALSE OR s_active THEN
    r_time := tx;
END_IF;

IF c_status = 0 AND tx - c_time > error_time AND c_enable AND NOT c_ready AND c_mode < 2 THEN (* Connect Timeout *)
	c_status := 255; (* Timeout-Fehler ausgeben *)
END_IF;
IF s_status = 0 AND tx - s_time > error_time AND s_active THEN (* Send Timeout *)
	s_status := 255; (* Timeout-Fehler ausgeben *)
END_IF;
IF r_status = 0 AND tx - r_time > error_time AND c_ready AND IP_C.R_OBSERVE THEN (* Receive Timeout *)
	r_status := 255; (* Timeout-Fehler ausgeben *)
END_IF;

IP_C.ERROR := OSCAT_BASIC.DWORD_OF_BYTE(c_status,s_status,r_status,0);

(* ---------------------- Debug-Message ----------------------------*)
(*IF  _debug_enable THEN
	IF (IP_C.ERROR <> debug_lasterror) THEN
		LOG_CL.NEW_MSG := 'IP_ERROR ~1 -IP_ID ~6';
		LOG_CL.PRINTF[1] := DWORD_TO_STRING(IP_C.ERROR);
		LOG_CL.PRINTF[6] := INT_TO_STRING(debug_ID);
		LOG_MSG();

		debug_lasterror := IP_C.ERROR;
		debug_last_id := debug_ID;
	END_IF;
END_IF;*)
(* -----------------------------------------------------------------------------*)

(* revision history
ks	18. oct. 2008	rev 1.0
	original version
ks	05. jan. 2009	rev 1.1
	zusätzlichen Parameter "Port" hinzugefügt, und Code bei Verbindungsaufbau angepasst
ks	28. feb. 2009	rev 1.2
	connect code angepasst und verkürzung der zyklen bei Send
ks	20. oct. 2009	rev 1.3
	network_buffer basis 0
ks	06. dec. 2009	rev 1.4
	receive_count MAILBOX[1]
ks	10. nov. 2010	rev 1.5
	send_pause MAILBOX[2] and receive_pause MAILBOX[3] 
ks	07. feb. 2011	rev 1.6
	Fehlercode 253 bei Remote-Verbindungsabbruch
ks	30. may. 2011	rev 1.7
	Fehler bei Empfangsüberwachung
ks	26. july. 2011	rev 1.8
	Error-Reset, diverse optimierungen 
ks	26. feb. 2012	rev 1.9
	Code-Optimierung für höhere kompatibilität

*)
END_FUNCTION_BLOCK
