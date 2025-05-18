FUNCTION_BLOCK IRTRANS_DECODE
VAR_IN_OUT
    IP_C : IP_Control; // IP_Control Verwaltungsstruktur
    R_BUF : NETWORK_BUFFER_SHORT;
END_VAR

VAR_OUTPUT
    CMD : BOOL;
    DEV : STRING;
    KEY : STRING;
    ERROR : BOOL;
END_VAR

VAR
    i : INT;
    stop : INT;
    pt : POINTER TO BYTE;
    buffer : POINTER TO ARRAY[0..32000] OF BYTE;
    size : UINT;
    z : BYTE;
END_VAR
size := R_BUF.SIZE;

(* do nothing if the buffer is empty *)
IF size > 0 AND IP_C.MAILBOX[1] <> z THEN
	i := 0;
	error := TRUE;
	stop := size - 2;
	(* read characters before first comma and store in device *)
	pt := ADR(dev);
	buffer := ADR(R_BUF.BUFFER);
	WHILE i < stop DO
		IF buffer^[i] = 44 THEN EXIT; END_IF;
		pt^ := buffer^[i];
		pt := pt + 1;
		i := i + 1;
	END_WHILE;
	(* terminate the device string *)
	pt^ := 0;
	(* read after first comma and store in key *)
	i := i + 1;
	pt := ADR(key);
	WHILE i < stop DO
		pt^ := buffer^[i];
		pt := pt + 1;
		i := i + 1;
	END_WHILE;
	(* terminate the key string *)
	pt^:= 0;
	(* check if last two characters are $R$N 13, 10 *)
	IF buffer^[i] <> 13 AND buffer^[i+1] <> 10 THEN	RETURN;	END_IF;
	(* check if device and key are present *)
	IF LEN(dev) = 0 OR LEN(key) = 0 THEN RETURN; END_IF;
	(* a valid transmission has been received *)
	cmd := TRUE;
	error := FALSE;
ELSE
	error := FALSE;
	cmd := FALSE;
END_IF;

z := IP_C.MAILBOX[1];

(* revision history
ks	18. oct. 2008	rev 1.0
	original version
ks	20. oct. 2009	rev 1.1
	Networkbuffer und Buffer-funktionen auf Basis 0
ks	06. dec. 2009	rev 1.2
	Codeanpassung für IRTRANS_SERVER
*)
END_FUNCTION_BLOCK
