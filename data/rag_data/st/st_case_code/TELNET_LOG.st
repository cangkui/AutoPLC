FUNCTION_BLOCK TELNET_LOG
VAR_IN_OUT
    IP_C : IP_C; // Derived Type
    S_BUF : NETWORK_BUFFER; // Derived Type
    LOG_CL : LOG_CONTROL; // Derived Type
END_VAR

VAR_INPUT
    ENABLE : BOOL;
    OPTION : BYTE := BYTE#16#8C; // no flush buffer (7), CR&LF (3), Colors (2), AUTOWRAP (1), Screen-Init (0)
    PORT : WORD := 23; // Port Number
END_VAR

VAR_OUTPUT
    READY : BOOL;
END_VAR

VAR
    TELNET_PRINT : TELNET_PRINT; // Derived Type
    done : BOOL;
    init : BOOL;
    watchdog : TON; // Derived Type
    ci : INT;
    pi : INT;
    pi_last : INT;
    send : BOOL;
END_VAR
IF NOT init THEN
	init := TRUE;
	IF OPTION > 0 THEN
		TELNET_PRINT.OPTION := OPTION;
	END_IF;
	IF PORT > 0 THEN (* bei portvorgabe wird der standardport überschrieben *)
		TELNET_PRINT.PORT := PORT;
	END_IF;
  	WATCHDOG.PT := T#1ms;
END_IF;

WATCHDOG(IN:=FALSE); (* Timer stoppen *)

ci := LOG_CL.IDX;

(* CONNECT_OFF Flanke auswerten *)
IF IP_C.C_STATE = BYTE#1 THEN
	pi := 0;
	init := FALSE;
END_IF;

REPEAT
	IF pi <> ci AND IP_C.C_STATE > BYTE#127 THEN
		IF pi = 0 AND LOG_CL.RING_MODE THEN pi := ci; END_IF;
		pi_last := pi;
		pi := pi + 1;
		IF pi > LOG_CL.SIZE THEN pi := 1; END_IF;
		SEND := TRUE;
	ELSE
		SEND := FALSE;
	END_IF;

	TELNET_PRINT(	IP_C := IP_C,
					S_BUF := S_BUF,
					ENABLE := ENABLE,
					SEND := SEND,
					TEXT := LOG_CL.MSG[pi],
					BACK_COLOR := DWORD_TO_BYTE(SHL(LOG_CL.MSG_OPTION[pi],8)),
					FRONT_COLOR := DWORD_TO_BYTE(LOG_CL.MSG_OPTION[pi] ));
	READY := TELNET_PRINT.READY;
	done := TELNET_PRINT.DONE;

	IF SEND AND NOT done THEN
		pi := pi_last;
	END_IF;

	WATCHDOG(IN:=TRUE);

UNTIL (WATCHDOG.Q OR NOT done )
END_REPEAT;

(* revision history
ks	14. aug. 2009	rev 1.0
	original version

ks	03. dec. 2011	rev 1.1
	neue parameter und initialisierung optimiert

*)
END_FUNCTION_BLOCK
