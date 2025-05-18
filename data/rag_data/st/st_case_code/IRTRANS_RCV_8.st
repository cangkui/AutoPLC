FUNCTION_BLOCK IRTRANS_RCV_8
VAR_INPUT
    dev_code : STRING;
    key_code_0 : STRING;
    key_code_1 : STRING;
    key_code_2 : STRING;
    key_code_3 : STRING;
    key_code_4 : STRING;
    key_code_5 : STRING;
    key_code_6 : STRING;
    key_code_7 : STRING;
    CMD : BOOL;
END_VAR

VAR_IN_OUT
    DEV : STRING;
    KEY : STRING;
END_VAR

VAR_OUTPUT
    Q0 : BOOL;
    Q1 : BOOL;
    Q2 : BOOL;
    Q3 : BOOL;
    Q4 : BOOL;
    Q5 : BOOL;
    Q6 : BOOL;
    Q7 : BOOL;
END_VAR

VAR_TEMP
    decode : BOOL;
END_VAR
decode := cmd AND dev_code = dev;

Q0 := key = key_code_0 AND decode;
Q1 := key = key_code_1 AND decode;
Q2 := key = key_code_2 AND decode;
Q3 := key = key_code_3 AND decode;
Q4 := key = key_code_4 AND decode;
Q5 := key = key_code_5 AND decode;
Q6 := key = key_code_6 AND decode;
Q7 := key = key_code_7 AND decode;


(* revision history
hm	2. jan 2008		rev 1.0
	original version

hm	17. oct 2008	rev 1.1
	improved performance

hm	19. oct. 2008	rev 1.2
	changed input device_code to dev_code and device to dev for compatibility reasons

hm	06. dec. 2008	rev 1.3
	IRTRANS_1 renamed to IRTRANS_RCV_8

*)
END_FUNCTION_BLOCK
