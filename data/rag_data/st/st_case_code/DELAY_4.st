FUNCTION_BLOCK DELAY_4
VAR_INPUT
    in : REAL;
END_VAR

VAR_OUTPUT
    out1 : REAL;
    out2 : REAL;
    out3 : REAL;
    out4 : REAL;
END_VAR

VAR
    temp : REAL;
END_VAR
out4 := out3;
out3 := out2;
out2 := out1;
out1 := temp;
temp := in;

(* revision history

hm 19.1.2007	rev 1.1
	added variable temp to add 1 delay  for q1

*)
END_FUNCTION_BLOCK
