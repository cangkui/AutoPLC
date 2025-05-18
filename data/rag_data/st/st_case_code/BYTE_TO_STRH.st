FUNCTION BYTE_TO_STRH : string
VAR_INPUT
    IN : BYTE; // Input variable
END_VAR

VAR
    temp : BYTE; // Local variable
    PT : POINTER TO BYTE; // Pointer to a BYTE
END_VAR
(* read pointer to output string *)
PT := ADR(BYTE_TO_STRH);
(* calculate high order hex value *)
temp := SHR(in,4);
(* convert value to hex character *)
IF temp <= 9 THEN temp := temp + 48; ELSE temp := temp + 55; END_IF;
(* write friat character to output string *)
PT^ := temp;
(* calculate low order hex value *)
temp := in AND 2#00001111;
IF temp <= 9 THEN temp := temp + 48; ELSE temp := temp + 55; END_IF;
(* increment pointer and wirte low order character *)
pt := pt + 1;
pt^ := temp;
(* set pointer at the end of the output string and close the string with 0 *)
pt := pt + 1;
pt^:= 0;


(* code before rev 1.2
FOR i := 1 TO 2 DO
	X := in AND 2#1111;
	IF X <= 9 THEN X := X + 48; ELSE X := X + 55; END_IF;
	Cx := CHR(X);
	temp := CONCAT(Cx, temp);
	in := SHR(in,4);
END_FOR;
BYTE_TO_STRH := temp;
*)

(* revision history

hm	9.6.2007		rev 1.0		
	original version 

hm	11.9.2007		rev 1.1		
	changed coding for compatibility with twincat, under twincat concat cannot have a function as argument.	

hm	15 dec 2007		REV 1.2
	changed code for higher performance

hm	29. mar. 2008	rev 1.3
	changed STRING to STRING(2)
*)
END_FUNCTION
