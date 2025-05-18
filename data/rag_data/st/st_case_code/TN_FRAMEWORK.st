FUNCTION_BLOCK TN_FRAMEWORK
VAR
    FB_IP_CONTROL : IP_CONTROL;  
    FB_TN_INPUT_CONTROL : TN_INPUT_CONTROL;  
    FB_TN_INPUT_MENU_BAR : TN_INPUT_MENU_BAR;  
    FB_TN_RECEIVE : TN_RECEIVE;  
    FB_TN_SEND_ROWS : TN_SEND_ROWS;  
    t : TON;  
END_VAR

VAR_IN_OUT
    us_TN_INPUT_CONTROL : us_TN_INPUT_CONTROL;  
    us_TN_SCREEN : us_TN_SCREEN;  
    us_TN_MENU : us_TN_MENU;  
    S_BUF : NETWORK_BUFFER;  
    R_BUF : NETWORK_BUFFER;  
    IP_C : IP_C;  
END_VAR_IN_OUT

VAR_INPUT
    PORT : WORD;  
END_VAR_INPUT
(* Initialisierung / Freigeben der Verbindung *)
IF (NOT IP_C.C_ENABLE) THEN

	(* IP Datenverkehr einrichten *)
	IP_C.C_PORT := SEL(PORT=BYTE#0,PORT,WORD#23); (* Portnummer eintragen *)
	IP_C.C_IP := DWORD#00; (* IP eintragen *)
	IP_C.C_MODE := BYTE#4; (* Mode: TCP+PASSIV+PORT *)
	IP_C.TIME_RESET := TRUE; (* Zeitueberwachung rücksetzen *)
	IP_C.C_ENABLE := TRUE; (* Verbindungsaufbau freigeben *)
	IP_C.R_OBSERVE := FALSE; (* keine Empfangsueberwachung *)

END_IF;

t(IN:= IP_C.ERROR > DWORD#0, PT:=T#5s);
IF t.Q THEN
	IP_C.TIME_RESET := TRUE; (* Error Reset *)
END_IF;

(* Telnet-Menu-Bar *)
FB_TN_INPUT_MENU_BAR(
	Xus_TN_MENU := us_TN_MENU,
	Xus_TN_SCREEN := us_TN_SCREEN);

(* Telnet-Input-Control *)
FB_TN_INPUT_CONTROL(
	Xus_TN_SCREEN := us_TN_SCREEN,
	Xus_TN_INPUT_CONTROL := us_TN_INPUT_CONTROL);

(* Telnet-Data-Send Line-Updates *)
FB_TN_SEND_ROWS(
	IP_C :=IP_C,
	S_BUF :=S_BUF,
	Xus_TN_SCREEN :=us_TN_SCREEN,
	S_BUF_SIZE :=SIZEOF(S_BUF.BUFFER));

(* Telnet-Server *)
FB_IP_CONTROL(
	IP_C :=IP_C,
	S_BUF :=S_BUF,
	R_BUF :=R_BUF,
	IP :=DWORD#00,
	PORT :=WORD#00,
	TIME_OUT :=T#2s);

(* Telnet-Data-Receive *)
FB_TN_RECEIVE(
	R_BUF := R_BUF,
	Xus_TN_SCREEN := us_TN_SCREEN);

(* revision history
ks	10. mai. 2010	rev 1.0
	original version

ks	17. may. 2011	rev 1.1
	automatischer re-connect

ks	05. may. 2012	rev 1.2
	neue parameter PORT
*)
END_FUNCTION_BLOCK
