FUNCTION_BLOCK READ_HTTP
VAR_INPUT
    SIZE : UINT;                    // Input variable of type UINT
    POS : INT;                      // Input variable of type INT
    SRC : STRING;                   // Input variable of type STRING
END_VAR

VAR_IN_OUT
    PT : NW_BUF_LONG;               // InOut variable of derived type NW_BUF_LONG
END_VAR

VAR_OUTPUT
    READ_HTTP : STRING[OSCAT_BASIC.STRING_LENGTH]; // Output variable of type STRING with defined length
END_VAR

VAR
    index : INT;                   // Local variable of type INT
    s1 : INT;                      // Local variable of type INT
    e1 : INT;                      // Local variable of type INT
    stop : INT;                    // Local variable of type INT
END_VAR
stop := UINT_TO_INT(SIZE) - 1;

IF LEN(SRC) = 0 THEN
	(* wenn kein Suchbegriff vorhanden ist, dann den HTTP-STATUS auswerten *)
	SRC := 'HTTP/'; (* neuer Suchbegriff *)
ELSE
	(* gesuchten Parametertext ergänzen *)
	SRC := CONCAT(SRC,': ');
END_IF;

s1 := OSCAT_BASIC.BUFFER_SEARCH(pt:=ADR(PT),SIZE:=SIZE,STR:=SRC,POS:=POS,IGN:=TRUE);

IF s1 >= 0 THEN (* Text found *)
	e1 := s1;

	(* search start of line *)
	index := s1 - 1;
	WHILE (index > 0) DO
		IF pt[index] < BYTE#32 THEN EXIT; END_IF;
		s1 := s1 - 1;
		index := index - 1;
	END_WHILE;

	(* search end of line *)
	index := e1 + 1;
	WHILE (index <= stop) DO
		IF pt[index] < BYTE#32 THEN EXIT; END_IF;
		e1 := e1 + 1;
		index := index + 1;
	END_WHILE;

	s1 := s1 + LEN(SRC); (* 'suchbegriff: ' nicht mitkopieren *)

	(* copy string *)
	READ_HTTP := OSCAT_BASIC.BUFFER_TO_STRING(PT:=ADR(PT),SIZE:=SIZE,START:=s1,STOP:=e1);
ELSE
	READ_HTTP := '';
END_IF;

(* revision history

ks	18. oct. 2008	rev 1.0
	original version

ks	20. oct. 2009	rev 1.1
	Networkbuffer und Buffer-funktionen auf Basis 0

*)
END_FUNCTION_BLOCK
