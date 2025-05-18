FUNCTION_BLOCK SIGNAL_4
VAR_INPUT
    IN1 : BOOL;
    IN2 : BOOL;
    IN3 : BOOL;
    IN4 : BOOL;
    TS : TIME;
    S1 : BYTE := 16#FF;  // Initial value set as 2#11111111
    S2 : BYTE := 16#F0;  // Initial value set as 2#11110000
    S3 : BYTE := 16#AA;  // Initial value set as 2#10101010
    S4 : BYTE := 16#A0;  // Initial value set as 2#10100000
END_VAR

VAR_OUTPUT
    Q : BOOL;
END_VAR

VAR
    sig : SIGNAL;  // Local variable of derived type SIGNAL
END_VAR
(* an alarm is present read system time first *)
(* check if an alarm is present if yes set sig to the alarm pattern otherwise exit the routine *)
IF in1 THEN
	sig(in := TRUE, sig := s1, TS := TS);
ELSIF in2 THEN
	sig(in := TRUE, sig := s2, TS := TS);
ELSIF in3 THEN
	sig(in := TRUE, sig := s3, TS := TS);
ELSIF in4 THEN
	sig(in := TRUE, sig := s4, TS := TS);
ELSE
	sig(in := FALSE);
END_IF;

(* set the output *)
Q := sig.Q;


(* revision history
hm	13.12.2007		rev 1.0
	original version

*)
END_FUNCTION_BLOCK
