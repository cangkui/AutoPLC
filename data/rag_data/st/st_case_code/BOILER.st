FUNCTION_BLOCK BOILER
VAR_INPUT
    T_upper : REAL;
    T_lower : REAL;
    Pressure : BOOL := TRUE;
    enable : BOOL := TRUE;
    Req_1 : BOOL;
    Req_2 : BOOL;
    Boost : BOOL;
    t_upper_min : REAL := 50.0;
    t_upper_max : REAL := 60.0;
    T_lower_enable : BOOL;
    T_lower_max : REAL := 60.0;
    T_request_1 : REAL := 70.0;
    T_request_2 : REAL := 50.0;
    T_request_hys : REAL := 5.0;
    T_protect_high : REAL := 80.0;
    T_protect_low : REAL := 10.0;
END_VAR_INPUT

VAR_OUTPUT
    Heat : BOOL;
    error : BOOL;
    status : BYTE;
END_VAR_OUTPUT

VAR
    edge : BOOL;
    boost_mode : BOOL;
    flag_0 : BOOL;
    flag_1 : BOOL;
    flag_2 : BOOL;
END_VAR
(* read sensors and check for valid data *)
IF t_upper > t_protect_high THEN
	status := 1;
	heat := FALSE;
	error := TRUE;
ELSIF t_upper < t_protect_low THEN
	status := 2;
	heat :=	TRUE;
	error := TRUE;
ELSIF t_lower > T_protect_high AND t_lower_enable THEN
	status := 3;
	heat := FALSE;
	error := TRUE;
ELSIF t_lower < t_protect_low AND t_lower_enable THEN
	status := 4;
	heat := TRUE;
	error := TRUE;
ELSIF NOT pressure THEN
	status := 5;
	heat := FALSE;
	error := TRUE;
ELSIF req_1 OR req_2 OR enable OR boost THEN
	error := FALSE;

	(* determine if heat needs to be turned on *)
	IF boost AND NOT edge AND t_upper < t_upper_max THEN
		status := 101;
		heat := TRUE;
		boost_mode := TRUE;
	ELSIF enable AND t_upper < T_upper_min THEN
		status := 102;
		heat := TRUE;
	ELSIF req_1 AND t_upper < T_request_1 THEN
		status := 103;
		heat := TRUE;
	ELSIF req_2 AND t_upper < t_request_2 THEN
		status := 104;
		heat := TRUE;
	END_IF;

	(* determine the shut off temperature *)
	IF heat THEN
		IF (enable OR boost_mode) THEN
			flag_0 := TRUE;
			IF T_lower_enable AND T_lower > T_lower_max THEN
				flag_0 := boost_mode := FALSE;
			ELSIF NOT T_lower_enable AND T_upper > T_upper_max THEN
				flag_0 := boost_mode := FALSE;
			END_IF;
		ELSE
			flag_0 := FALSE;
		END_IF;
		flag_1 := (req_1 AND T_upper > T_request_1 + T_request_hys) AND req_1;
		flag_2 := (req_2 AND T_upper > T_request_2 + T_request_hys) AND req_2;

		(* shut off heat if no longer needed *)
		heat := flag_0 OR flag_1 OR flag_2;
		IF heat = FALSE THEN status := 100; END_IF;
	END_IF;
ELSE
	status := 100;
	heat := FALSE;
	error := FALSE;
END_IF;
edge := boost;


(* revision history
hm	27. feb. 2007	rev 1.0
	original version

hm	16. oct. 2008	rev 1.1
	improved performance

hm	13. mar. 2009	rev 1.2
	real constants updated to new systax using dot
	replaced double assignments
*)
END_FUNCTION_BLOCK
