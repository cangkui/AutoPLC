FUNCTION SHA1_MD5_DEMO
VAR
    text: STRING(OSCAT_BASIC.STRING_LENGTH) := '<4036.1076324381@guildenstern.zeitform.de>';
    MD5_STR: MD5_STR;
    SHA1_STR: SHA1_STR;
    md5_hash: ARRAY[0..15] OF BYTE;
    sha1_hash: ARRAY[0..19] OF BYTE;
    start1: BOOL := TRUE;
    start2: BOOL := TRUE;
    done1: BOOL;
    done2: BOOL;
    md5_strh: STRING(32);
    sha1_strh: STRING(40);
END_VAR

END_FUNCTION
