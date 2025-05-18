FUNCTION_BLOCK MANUAL_1
VAR_INPUT
    IN : BOOL;
    MAN : BOOL;
    M_I : BOOL;
    SET : BOOL;
    RST : BOOL;
END_VAR

VAR_OUTPUT
    Q : BOOL;
    STATUS : BYTE;
END_VAR

VAR
    S_edge : BOOL;
    r_edge : BOOL;
    edge : BOOL;
END_VAR
IF NOT man THEN
	Q := IN;
	STATUS := 100;
	edge := FALSE;
ELSIF NOT s_edge AND set THEN
	Q := TRUE;
	edge := TRUE;
	status := 101;
ELSIF NOT r_edge AND rst THEN
	Q := FALSE;
	edge := TRUE;
	status := 102;
ELSIF NOT edge THEN
	Q := M_I;
	status := 103;
END_IF;

(* remember levels of manual signals *)
s_edge := SET;
r_edge := RST;



(* revision history
hm	17. jun 2008	rev 1.0
	original version

hm	17. oct 2008	rev 1.1
	deleted unnecessary variable m_edge

hm	14. mar. 2009	rev 1.2
	replaced double assignments

*)
END_FUNCTION_BLOCK
