FUNCTION OFFSET2 : REAL
VAR_INPUT
    X : REAL;  
    O1 : BOOL;  
    O2 : BOOL;  
    O3 : BOOL;  
    O4 : BOOL;  
    D : BOOL;  
    Offset_1 : REAL;  
    Offset_2 : REAL;  
    Offset_3 : REAL;  
    Offset_4 : REAL;  
    default : REAL;  
END_VAR
IF D THEN OFFSET2 := default; ELSE OFFSET2 := X; END_IF;
IF O4 THEN OFFSET2 := OFFSET2 + offset_4;
ELSIF O3 THEN OFFSET2 := OFFSET2 + offset_3;
ELSIF O2 THEN OFFSET2 := OFFSET2 + offset_2;
ELSIF O1 THEN OFFSET2 := OFFSET2 + offset_1;
END_IF;
END_FUNCTION
