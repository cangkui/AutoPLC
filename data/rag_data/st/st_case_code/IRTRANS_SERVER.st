FUNCTION_BLOCK IRTRANS_SERVER
VAR_IN_OUT
    IP_C : IP_C; // IP_Control Verwaltungsstruktur
    S_BUF : NETWORK_BUFFER_SHORT;
    R_BUF : NETWORK_BUFFER_SHORT;
END_VAR

VAR_INPUT
    UDP_TCP : BOOL;
END_VAR

VAR_OUTPUT
    S_ENABLE : BOOL;
    R_ENABLE : BOOL;
    ERROR : DWORD;
END_VAR

VAR
    t : TON;
END_VAR
IF NOT IP_C.C_ENABLE THEN
	(* IP Datenverkehr einrichten *)
	IP_C.C_PORT := WORD#0; (* Portnummer eintragen *)
	IP_C.C_IP := DWORD#00; (* IP eintragen *)
	IP_C.C_MODE := SEL(UDP_TCP,5,4); (* 4 = Mode: TCP+PASSIV+PORT / 5 = Mode: UDP+PASSIV+PORT*)
	IP_C.C_ENABLE := TRUE; (* Verbindungsaufbau freigeben *)
	IP_C.R_OBSERVE := FALSE; (* Datenempfang ueberwachen *)
	IP_C.TIME_RESET := TRUE; (* Zeitueberwachung rücksetzen *)
	S_BUF.SIZE := UINT#0; (* Sendedaten rücksetzen *)
	R_BUF.SIZE := UINT#0; (* Empfangsdaten rücksetzen *)
END_IF;

R_ENABLE := IP_C.C_STATE > 127;
S_ENABLE := R_ENABLE AND (IP_C.MAILBOX[1] > 0 OR IP_C.C_MODE <> 5);
ERROR := IP_C.ERROR;

t(IN:= IP_C.ERROR > 0, PT:=T#5s);
IF t.Q THEN
	IP_C.TIME_RESET := TRUE; (* Error Reset *)
END_IF;

(* revision history
ks	6. dec 2009		rev 1.0
	original version

ks	17. may 2011		rev 1.1
	neue Fehlerausgabe und Empfangsbuffer rücksetzen
*)
END_FUNCTION_BLOCK
