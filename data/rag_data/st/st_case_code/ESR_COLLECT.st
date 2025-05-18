FUNCTION_BLOCK ESR_COLLECT
VAR_INPUT
    ESR_0 : ARRAY[0..3] OF esr_data; 
    ESR_1 : ARRAY[0..3] OF esr_data; 
    ESR_2 : ARRAY[0..3] OF esr_data; 
    ESR_3 : ARRAY[0..3] OF esr_data; 
    ESR_4 : ARRAY[0..3] OF esr_data; 
    ESR_5 : ARRAY[0..3] OF esr_data; 
    ESR_6 : ARRAY[0..3] OF esr_data; 
    ESR_7 : ARRAY[0..3] OF esr_data; 
    rst : BOOL; 
END_VAR

VAR_IN_OUT
    pos : INT; 
END_VAR

VAR_OUTPUT
    ESR_OUT : ARRAY[0..31] OF esr_data; 
END_VAR

VAR CONSTANT
    max_in : INT := 3; // max limit of input array 
    max_out : INT := 32; // number of elements in array 
END_VAR

VAR
    cnt : INT := -1; 
END_VAR
IF rst OR cnt < 0 THEN
	pos := -1;
ELSE
	FOR cnt := 0 TO max_in DO
	IF esr_0[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_0[cnt]; END_IF;
	IF esr_1[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_1[cnt]; END_IF;
	IF esr_2[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_2[cnt]; END_IF;
	IF esr_3[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_3[cnt]; END_IF;
	IF esr_4[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_4[cnt]; END_IF;
	IF esr_5[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_5[cnt]; END_IF;
	IF esr_6[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_6[cnt]; END_IF;
	IF esr_7[cnt].typ > 0 THEN pos := INC1(pos, max_out); esr_out[pos] := esr_7[cnt]; END_IF;
   END_FOR;
END_IF;


(* revision history
hm	26.jan 2007	rev 1.0
	original version

hm	8. dec 2007	rev 1.1
	added reset input

ks	27. oct. 2008	rev 1.2
	optimized code for performance

ks	12. nov. 2009	rev 1.3
	output pos was not pointing to last value	

hm	1. dec. 2009	rev 1.4
	changed pos to be I/O
	reduced output array size to 32 elements
*)
END_FUNCTION_BLOCK
