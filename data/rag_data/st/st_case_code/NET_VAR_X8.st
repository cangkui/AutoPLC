FUNCTION_BLOCK NET_VAR_X8
VAR_IN_OUT
    X : NET_VAR_DATA;  
END_VAR

VAR_INPUT
    IN_REAL1 : REAL;  
    IN_REAL2 : REAL;  
    IN_DINT1 : DINT;  
    IN_DINT2 : DINT;  
    IN_UDINT1 : UDINT;  
    IN_UDINT2 : UDINT;  
    IN_DWORD1 : DWORD;  
    IN_DWORD2 : DWORD;  
END_VAR_INPUT

VAR_OUTPUT
    OUT_REAL1 : REAL;  
    OUT_REAL2 : REAL;  
    OUT_DINT1 : DINT;  
    OUT_DINT2 : DINT;  
    OUT_UDINT1 : UDINT;  
    OUT_UDINT2 : UDINT;  
    OUT_DWORD1 : DWORD;  
    OUT_DWORD2 : DWORD;  
    ID : BYTE;  
END_VAR_OUTPUT

VAR
    init : BOOL;  
    i : INT;  
    i2 : INT;  
    index : INT;  
    index2 : INT;  
    ra : ARRAY[1..8] OF DWORD;  
END_VAR
IF NOT init THEN
	init := TRUE;
	X.ID_MAX := X.ID_MAX + USINT#1;
	ID := USINT_TO_BYTE(X.ID_MAX);
END_IF;

IF X.STATE > BYTE#0 AND X.ERROR_ID = BYTE#0 THEN
	index := X.index; (* PCWORX Workaround *)
	IF INT_TO_UINT(index + 34) > X.BUF_SIZE THEN  (* ID (1) + TYPE (1) + 8 x DWORD (4) *)
		X.ERROR_ID := ID;
	ELSE
		IF X.STATE = BYTE#1 THEN (* Daten schreiben *)
			ra[1] := OSCAT_BASIC.REAL_TO_DW(X:=IN_REAL1);
			ra[2] := OSCAT_BASIC.REAL_TO_DW(X:=IN_REAL2);
			ra[3] := DINT_TO_DWORD(IN_DINT1);
			ra[4] := DINT_TO_DWORD(IN_DINT2);
			ra[5] := UDINT_TO_DWORD(IN_UDINT1);
			ra[6] := UDINT_TO_DWORD(IN_UDINT2);
			ra[7] := IN_DWORD1;
			ra[8] := IN_DWORD2;
			X.S_BUF.BUFFER[index] := ID; (* ID *)
			index := index + 1;
			X.S_BUF.BUFFER[index] := BYTE#21; (* TYPE X8 *)
			index := index + 1;
			FOR i2 := 1 TO 8 DO;
				FOR i := 1 TO 4 DO;
					X.S_BUF.BUFFER[index] := DWORD_TO_BYTE(ra[i2]);
					ra[i2] := ROR(ra[i2],8);
					index := index + 1;
				END_FOR;
			END_FOR;

		ELSIF X.STATE = BYTE#2 THEN (* Daten lesen *)
			index2 := index + 1; (* PCWORX Workaround *)
			IF X.R_BUF.BUFFER[index] <> ID OR X.R_BUF.BUFFER[index2] <> BYTE#21 THEN (* TYPE X8 *)
				X.ERROR_ID := ID;
			ELSE
				index := index + 2;
				FOR i2 := 1 TO 8 DO;
					ra[i2] := DWORD#0;
					FOR i := 1 TO 4 DO;
						ra[i2] := ra[i2] OR BYTE_TO_DWORD(X.R_BUF.BUFFER[index]);
						ra[i2] := ROR(ra[i2],8);
						index := index + 1;
					END_FOR;
				END_FOR;
				OUT_REAL1 := OSCAT_BASIC.DW_TO_REAL(X:=ra[1]);
				OUT_REAL2 := OSCAT_BASIC.DW_TO_REAL(X:=ra[2]);
				OUT_DINT1 := DWORD_TO_DINT(ra[3]);
				OUT_DINT2 := DWORD_TO_DINT(ra[4]);
				OUT_UDINT1 := DWORD_TO_UDINT(ra[5]);
				OUT_UDINT2 := DWORD_TO_UDINT(ra[6]);
				OUT_DWORD1 := ra[7];
				OUT_DWORD2 := ra[8];
			END_IF;
		END_IF;
		X.index := index;
	END_IF;
END_IF;

(* revision history
ks	15.may. 2011	rev 1.0
	original version

*)
END_FUNCTION_BLOCK
