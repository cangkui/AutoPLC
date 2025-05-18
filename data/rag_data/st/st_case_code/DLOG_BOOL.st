FUNCTION_BLOCK DLOG_BOOL
VAR_IN_OUT
    X : DLOG_DATA;
END_VAR

VAR_INPUT
    STATE : BOOL;
    OFF : STRING;
    ON : STRING;
    COLUMN : STRING[40];
    DELTA : BOOL;
END_VAR_INPUT

VAR
    UCB : UNI_CIRCULAR_BUFFER;
    id : WORD;
    last_delta : BOOL;
    state_last : BOOL;
END_VAR
CASE x.ADD_COM OF

01:	(* ADD INFO *)
	X.ID_MAX := X.ID_MAX + USINT#1;
	id := WORD#16#0601; (* Quelltype BOOL , Zieltype STRING *)
02:	(* ADD HEADER *)
	X.UCB.D_STRING := COLUMN;
	X.UCB.D_HEAD := id;
	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
03:	(* ADD DATA *)
	IF X.STORE_TYPE = BYTE#05 THEN (* MySQL-Modus *)
		X.UCB.D_STRING := SEL(state_last,'0','1');
	ELSE
		X.UCB.D_STRING := SEL(state_last,OFF,ON);
	END_IF;
	X.UCB.D_HEAD := id;

	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
	last_delta := STATE;
04:	(* ADD DATA REQ *)
	IF DELTA AND (STATE XOR last_delta) THEN
		X.ADD_DATA_REQ := TRUE;
		last_delta := STATE;
	END_IF;
END_CASE;
state_last := STATE;

(* revision history
ks	17. may. 2011	rev 1.0
	original version

ks	25. july.   2011	rev 1.1
	neue head information

ks	19. apr.   2012	rev 1.2
	code neu überarbeitet

*)
END_FUNCTION_BLOCK
