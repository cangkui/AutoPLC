FUNCTION_BLOCK DLOG_DINT
VAR_IN_OUT
    X : DLOG_DATA;        // Derived type
END_VAR

VAR_INPUT
    VALUE : DINT;        // Integer value
    COLUMN : STRING[40]; // String with length 40
    DELTA : DINT;        // Integer delta value
END_VAR

VAR
    UCB : UNI_CIRCULAR_BUFFER; // Derived type
    id : WORD;                 // Word integer
    delta_last : DINT;        // Last delta value
    value_last : DINT;        // Last value
END_VAR
CASE x.ADD_COM OF

01:	(* ADD INFO *)
	X.ID_MAX := X.ID_MAX + USINT#1;
	id := WORD#16#0801; (* Quelltype DINT , Zieltype STRING *)
02:	(* ADD HEADER *)
	X.UCB.D_STRING := COLUMN;
	X.UCB.D_HEAD := id;
	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
03:	(* ADD DATA *)
	X.UCB.D_STRING := DINT_TO_STRING(value_last);
	X.UCB.D_HEAD := id;
	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
	delta_last := value_last;
04:	(* ADD DATA REQ *)
	IF DELTA <> DINT#0 THEN
		IF VALUE <= (delta_last - DELTA) OR VALUE >= (delta_last + DELTA) THEN
			X.ADD_DATA_REQ := TRUE;
			delta_last := VALUE;
		END_IF;
	END_IF;
END_CASE;
value_last := VALUE;

(* revision history
ks	01. jan. 2011	rev 1.0
	original version

ks	17. may. 2011	rev 1.1
	DELTA Funktion integriert

ks	25. july.   2011	rev 1.2
	neue head information

ks	19. apr.   2012	rev 1.2
	code neu überarbeitet
*)
END_FUNCTION_BLOCK
