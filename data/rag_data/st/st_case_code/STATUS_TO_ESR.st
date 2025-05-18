FUNCTION STATUS_TO_ESR : esr_data
VAR_INPUT
    status : BYTE;
    adress : STRING[10];
    DT_in : DT;
    TS : TIME;
END_VAR
IF status < 100 THEN
	status_to_ESR.typ := 1;
ELSIF status < 200 THEN
	status_to_ESR.typ := 2;
ELSE
	status_to_ESR.typ := 3;
END_IF;
status_to_ESR.adress:= adress;
status_to_ESR.DS := DT_in;
status_to_ESR.TS := TS;
status_to_ESR.data[0] := status;
END_FUNCTION
