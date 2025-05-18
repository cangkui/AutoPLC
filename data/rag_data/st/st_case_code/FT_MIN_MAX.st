FUNCTION_BLOCK FT_MIN_MAX
VAR_INPUT
    in : REAL;
    rst : BOOL;
END_VAR

VAR_OUTPUT
    mx : REAL;
    mn : REAL;
END_VAR

VAR
    init : BOOL;
END_VAR
IF rst OR NOT init THEN
	mn := in;
	mx := in;
	init := TRUE;
ELSIF in < mn THEN mn := in;
ELSIF in > mx THEN mx := in;
END_IF;
END_FUNCTION_BLOCK
