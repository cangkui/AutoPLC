FUNCTION_BLOCK IRTRANS_SND_1
VAR_INPUT
    DEV_CODE : STRING;
    KEY_CODE : STRING;
    IN : BOOL;
    T_REPEAT : TIME;
END_VAR

VAR_IN_OUT
    IP_C : IP_C; // IP_Control Verwaltungsstruktur
    S_BUF : NETWORK_BUFFER_SHORT;
END_VAR

VAR_OUTPUT
    KEY : BYTE;
END_VAR

VAR
    skey : STRING;
    size : INT;
    str : STRING;
    t : TON;
    k : BYTE;
    lk : BYTE;
    d : BOOL;
END_VAR
d := IP_C.C_STATE > 127 AND (IP_C.MAILBOX[1] > 0 OR IP_C.C_MODE <> 5);
IF NOT d THEN RETURN; END_IF;

t(IN:= lk = k , PT:=T_REPEAT);

IF S_BUF.SIZE = 0 THEN
	IF IN THEN
		skey := KEY_CODE; k := 1;
	ELSE
		lk := 0;
		KEY := 0;
		RETURN; (* beenden wenn kein Key aktiv *)
	END_IF;

	IF lk <> k OR T.Q THEN (* anderer Key oder repeat wartezeit abgelaufen *)
		str := CONCAT('snd ',dev_code);
		str := CONCAT(str,',');
		str := CONCAT(str,skey);
		size := LEN(str);
		OSCAT_BASIC._STRING_TO_BUFFER(STR:=str, POS:=0, PT:=ADR(S_BUF.BUFFER), SIZE:=size);
		S_BUF.SIZE := size; (* Sendelänge eintragen *)
		KEY := k;
		lk := k; (* letzten Key speichern *)
		t(IN:=FALSE); (* Reset Time *)
	END_IF;
END_IF;

(* revision history
hm	6. dec 2009		rev 1.0
	original version
*)
END_FUNCTION_BLOCK
