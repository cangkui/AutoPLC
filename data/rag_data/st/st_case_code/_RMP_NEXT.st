FUNCTION_BLOCK _RMP_NEXT
VAR_INPUT
    E : BOOL := TRUE; // Initial value set to TRUE
    IN : BYTE;
    TR : TIME;
    TF : TIME;
    TL : TIME;
END_VAR

VAR_OUTPUT
    DIR : BOOL; // upwards = TRUE
    UP : BOOL;
    DN : BOOL;
END_VAR

VAR_IN_OUT
    OUT : BYTE;
END_VAR

VAR
    rmx : _RMP_B; // Derived type
    dirx : TREND_DW; // Derived type
    t_lock : TP; // Derived type
    xen : BOOL;
    xdir : BOOL;
END_VAR
dirx(X := in);

t_lock(in := FALSE, pt := TL);

IF dirx.TU AND (OUT < IN) THEN
	IF NOT xdir AND xen THEN t_lock(in := TRUE); END_IF;
	xen := TRUE;
	xdir := TRUE;
ELSIF dirx.TD AND (OUT > IN) THEN
	IF xdir AND xen THEN t_lock(in := TRUE); END_IF;
	xen := TRUE;
	xdir := FALSE;
ELSIF xen THEN
	IF (xdir AND (out >= in)) OR (NOT xdir AND (out <= in)) THEN
		xen := FALSE;
		IF tl > t#0s THEN t_lock(IN := TRUE); END_IF;
	END_IF;
END_IF;

IF NOT t_lock.Q AND xen THEN
		UP := XDIR;
		DIR := XDIR;
		DN := NOT XDIR;
ELSE
	UP := FALSE;
	DN := FALSE;
END_IF;

rmx(rmp := OUT, E := E AND (UP OR DN) , dir := DIR, tr := SEL(dir, TF, TR));


(* revison history
hm	23. nov. 2008	rev 1.0
	original version

hm	24. jan. 2009	rev 1.1
	deleted unused vars tmp1, tmp2

hm	20. feb. 2009	rev 1.2
	improved algorithm
	added TL

hm	9. mar. 2009	rev 1.3
	input E was ignored
	removed double assignments

hm	18. jul. 2009	rev 1.4
	improved performance

*)
END_FUNCTION_BLOCK
