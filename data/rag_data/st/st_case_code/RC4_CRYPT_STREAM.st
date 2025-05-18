FUNCTION_BLOCK RC4_CRYPT_STREAM
VAR_IN_OUT
    MODE : INT;
    KEY : STRING[40];
    BUF : ARRAY[0..63] OF BYTE;
    SIZE : UDINT;
END_VAR

VAR_OUTPUT
    POS : UDINT;
END_VAR

VAR
    sbox : ARRAY[0..255] OF BYTE;
    skey : ARRAY[0..255] OF BYTE;
    a : INT;
    b : INT;
    c : INT;
    d : USINT;
    e : USINT;
    end : UDINT;
    v : BYTE;
END_VAR
CASE MODE OF

1:	(* Initialisierung *)
	b := LEN(KEY) - 1;
	IF b < 0 OR SIZE < 1 THEN
		MODE := 3;
		RETURN;
	END_IF;

	FOR a := 0 TO b DO
		skey[a] := OSCAT_BASIC.CODE(KEY, a + 1);
		sbox[a] := INT_TO_BYTE(a);
	END_FOR;

	c := 0;
	FOR a := b + 1 TO 255 DO
		skey[a] := skey[c];
		sbox[a] := INT_TO_BYTE(a);
		c := c + 1;
		IF c > b THEN c := 0; END_IF;
	END_FOR;

	d := 0;
	FOR a := 0 TO 255 DO
		d := d + sbox[a] + skey[a];
		v := sbox[a];
		sbox[a] := sbox[d];
		sbox[d] := v;
	END_FOR;

	POS := 0;
	end := SIZE;
	SIZE := MIN(64,end);
	d := 0;
	e := 0;
	MODE := 2;

2:	(* Daten ver/entschlüsseln *)

	b := UDINT_TO_INT(SIZE) - 1;
	FOR a := 0 TO b DO
		d := d + 1;
		e := e + sbox[d];
		v := sbox[d];
		sbox[d] := sbox[e];
		sbox[e] := v;
		v := sbox[d] + sbox[e];
		BUF[a] :=BUF[a] XOR sbox[v];
	END_FOR;

	POS := POS + SIZE;
	SIZE := MIN(64,end - POS);

	IF SIZE = 0 THEN
		MODE := 3;
	END_IF;
END_CASE;

(* revision history

ks	9. Jan. 2010 rev 1.0
	original version

*)
END_FUNCTION_BLOCK
