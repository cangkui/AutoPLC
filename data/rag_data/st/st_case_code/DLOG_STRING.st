FUNCTION_BLOCK DLOG_STRING
VAR_IN_OUT
    X : DLOG_DATA;
END_VAR

VAR_INPUT
    STR : STRING;
    COLUMN : STRING[40];
END_VAR

VAR
    UCB : UNI_CIRCULAR_BUFFER;
    id : WORD;
    str_last : STRING;
END_VAR
CASE x.ADD_COM OF

01:	(* ADD INFO *)
	X.ID_MAX := X.ID_MAX + USINT#1;
	id := WORD#16#0101; (* Quelltype STRING , Zieltype STRING *)
02:	(* ADD HEADER *)
	X.UCB.D_STRING := COLUMN;
	X.UCB.D_HEAD := id;
	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
03:	(* ADD DATA *)
	X.UCB.D_STRING := str_last;
	X.UCB.D_HEAD := id;
	X.UCB.D_MODE := 1;
	UCB(DATA:=X.UCB); (* Daten eintragen *)
END_CASE;
str_last := STR;

(* revision history
ks	01. jan. 2011	rev 1.0
	original version

ks	17. may. 2011	rev 1.1
	COLUMN STRING(40)

ks	25. july.   2011	rev 1.2
	neue head information

ks	19. apr.   2012	rev 1.3
	code neu überarbeitet
*)
END_FUNCTION_BLOCK
