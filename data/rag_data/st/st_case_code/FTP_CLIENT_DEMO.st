FUNCTION FTP_CLIENT_DEMO
VAR
    FTP_CLIENT : FTP_CLIENT;  
    ftp_activate : BOOL;
    ftp_done : BOOL;
    ftp_busy : BOOL;
    ftp_error_c : DWORD;
    ftp_error_t : BYTE;
    ftp_url : STRING := 'ftp://581920:123456@oscat.atwebpages.com:2000/subordner/';
    ftp_filename : STRING := '0100kb.bin';
    ftp_download : BOOL := TRUE;
    ftp_activ : BOOL := FALSE;
    file_delete : BOOL := FALSE;
    ftp_timeout : TIME := TIME#30s0ms;
    zeit : TIME := TIME#30s0ms;
    stop : BOOL;
    start : BOOL := TRUE;
    reset : BOOL;
    error : DWORD;
    ok : DWORD;
    CLK_PRG_1 : oscatie_basic.CLK_PRG;
    COUNT_DR_1 : oscatie_basic.COUNT_DR;
    COUNT_DR_2 : oscatie_basic.COUNT_DR;
END_VAR

END_FUNCTION
