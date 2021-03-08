import ply.lex as lex
import feature_url
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenstring(data):
    # List of token names.   This is always required
    tokens = (
       'WGET',
        'KEY',
        'DBMS_PIPE',
        'S',
        'PASSWORD',
        'ONEVENT',
        'WHEN',
        'COMMAND',
        'HEAD',
        'PASSWD',
        'VBSCRIPT',
        'ALERT',
        'TITLE',
        'EMBED',
        'DOCUMENT',
        'PARTITION',
        'ON',
        'BIG',
        'RLIKE',
        'OFFSET',
        'HIGH_PRIORITY',
        'CONTENT',
        'DATALIST',
        'NUM',
        'FORM',
        'INTO',
        'SLEEP',
        'BODY',
        'U',
        'SUB',
        'IP',
        'RBRACKET',
        'DIV',
        'EXEC',
        'TD',
        'UPDATE',
        'FORMATION',
        'TILDE',
        'LIKE',
        'THEAD',
        'LOW_PRIORITY',
        'COMMA',
        'LBRACKET',
        'RUBY',
        'MAP',
        'EXISTS',
        'SIGN',
        'CURL',
        'VAR',
        'SCRIPT',
        'TBODY',
        'FOR',
        'UNION',
        'WORD',
        'DOCTYPE',
        'DELETE',
        'DOT',
        'SMALL',
        'BETWEEN',
        'SUBSTRING',
        'COL',
        'END',
        'POSTER',
        'ASC',
        'COLGROUP',
        'AREA',
        'MAKE',
        'FOOTER',
        'AWK',
        'PART',
        'AS',
        'ARTICLE',
        'TT',
        'CAPTION',
        'BLOCKQUOTE',
        'BASEFONT',
        'P',
        'HR',
        'SHARE',
        'ROLLUP',
        'DEL',
        'FRAME',
        'TIME',
        'PING6',
        'RPAREN',
        'JSCRIPT',
        'PLUS',
        'WHERE',
        'CREATE',
        'FILTER',
        'CASCADE',
        'LI',
        'TO',
        'DUMPFILE',
        'DESC',
        'IF',
        'PROGRESS',
        'SECTION',
        'BENCHMARK',
        'FIGCAPTION',
        'USING',
        'VIDEO',
        'OUTER',
        'JOIN',
        'FULL',
        'ASTER',
        'PS',
        'IS',
        'FROM',
        'SEMI',
        'DT',
        'PRE',
        'SET',
        'BDO',
        'HGROUP',
        'SOURCE',
        'TH',
        'ORDER',
        'OPTGROUP',
        'LIVESCRIPT',
        'DROP',
        'CHAR',
        'TEMPORARY',
        'COLLATE',
        'CAT',
        'RESTRICT',
        'META',
        'OBJECT',
        'TR',
        'HAVING',
        'RT',
        'CASE',
        'FONT',
        'IMG',
        'NAV',
        #'keyword_end',
        'NOT',
        'PROCEDURE',
        'XMLNS',
        'WAITFOR',
        'UL',
        'SELECT',
        'BINARY',
        'CODE',
        'TOP',
        'EQUCOMPARISON',
        'SQS',
        'BOOLEAN',
        'SUBSTR',
        'BIN',
        'FIELDSET',
        'SAMP',
        'SUP',
        'SIGN2',
        'DUPLICATE',
        'B',
        'AUDIO',
        'ANY',
        'BASE',
        'FRAMESET',
        'METER',
        'GROUP',
        'RP',
        'split',
        'PYTHON',
        'CANVAS',
        'DELAYED',
        'HEADER',
        'OPTION',
        'ACTION',
        'LS',
        'RECEIVE_MESSAGE',
        'WITH',
        'MINUSMINUS',
        'SRCDOC',
        'THEN',
        'MODE',
        'TRACK',
        'DD',
        'IFRAME',
        'IGNORE',
        'SPAN',
        'CONFIRM',
        'XLINK',
        'TABLE',
        'IN',
        'INNER',
        'OUTPUT',
        'ignore',
        'XMP',
        'BUTTON',
        'ADDRESS',
        'ALL',
        'NOSCRIPT',
        'STRONG',
        'NOFRAMES',
        'SOME',
        'LPAREN',
        'DELAY',
        'ACRONYM',
        'STRIKE',
        'DFN',
        'INTERVAL',
        'OL',
        'OUTFILE',
        'MARK',
        'BEGIN',
        'Q',
        'BDI',
        'CITE',
        'STYLE',
        'LABEL',
        'LEFT',
        'ISINDEX',
        'A',
        'HTML',
        'MSCRIPT',
        'APPLET',
        'DISTINCT',
        'INSERT',
        'INS',
        'DIR',
        'REGEXP',
        'ATTRIBUTENAME',
        'SRC',
        'TEXTAREA',
        'DATATYPE',
        'ASIDE',
        'BR',
        'ASCII',
        #'keyword_start',
        'QUICK',
        'EXCLA',
        'DECLARE',
        'RIGHT',
        'KEYGEN',
        'PROMPT',
        'BY',
        'GOTO',
        'LIMIT',
        'SUMMARY',
        'ELSE',
        'CENTER',
        'DATABASE',
        'MINUS',
        'NETSTAT',
        'FIGURE',
        'TFOOT',
        'PARAM',
        'EM',
        'LEGEND',
        'INPUT',
        'KBD',
        'HASH',
        'DETAILS',
        'REPLACE',
        'VALUES',
        'HREF',
        'JAVASCRIPT',
        'PING',
        'FUNCTION',
        'MENU',
        'LINK',
        'H1',
        'ABBR',
        'LOCK',
        'DL',

    )

    # Regular expression rules for simple tokens
    t_BASE   = r'BASE'
    t_NETSTAT   = r'NETSTAT'
    t_TH   = r'TH'
    t_ORDER   = r'ORDER'
    t_BOOLEAN   = r'BOOLEAN'
    t_CONTENT   = r'CONTENT'
    t_APPLET   = r'APPLET'
    t_ADDRESS   = r'ADDRESS'
    t_TABLE   = r'TABLE'
    t_DL   = r'DL'
    t_LIVESCRIPT   = r'LIVESCRIPT'
    t_JAVASCRIPT   = r'JAVASCRIPT'
    t_BR   = r'BR'
    t_HREF   = r'HREF'
    t_ACRONYM   = r'ACRONYM'
    t_COLLATE   = r'COLLATE'
    t_FOR   = r'FOR'
    t_TBODY   = r'TBODY'
    t_COLGROUP   = r'COLGROUP'
    t_SUBSTR   = r'SUBSTR'
    t_SHARE   = r'SHARE'
    t_EXEC   = r'EXEC'
    t_ignore   = r'.'
    t_FORM   = r'FORM'
    t_ASTER   = r'\*'
    t_RIGHT   = r'RIGHT'
    t_OL   = r'OL'
    t_BODY   = r'BODY'
    t_HIGH_PRIORITY   = r'HIGH_PRIORITY'
    t_STRONG   = r'STRONG'
    t_SRCDOC   = r'SRCDOC'
    t_UL   = r'UL'
    t_PART   = r'PART'
    t_TO   = r'TO'
    t_LOCK   = r'LOCK'
    t_TT   = r'TT'
    t_AUDIO   = r'AUDIO'
    t_OPTION   = r'OPTION'
    t_FUNCTION   = r'FUNCTION'
    t_KBD   = r'KBD'
    t_RECEIVE_MESSAGE   = r'RECEIVE_MESSAGE'
    t_EM   = r'EM'
    t_DATATYPE   = r'DATA\:TEXT/XML'
    t_PROGRESS   = r'PROGRESS'
    t_RUBY   = r'RUBY'
    t_THEN   = r'THEN'
    t_BIG   = r'BIG'
    t_DBMS_PIPE   = r'DBMS_PIPE'
    t_LOW_PRIORITY   = r'LOW_PRIORITY'
    t_FIELDSET   = r'FIELDSET'
    t_FULL   = r'FULL'
    t_OUTFILE   = r'OUTFILE'
    t_USING   = r'USING'
    t_DELAYED   = r'DELAYED'
    t_JOIN   = r'JOIN'
    t_WHERE   = r'WHERE'
    t_RP   = r'RP'
    t_OPTGROUP   = r'OPTGROUP'
    t_ASCII   = r'ASCII'
    t_NOFRAMES   = r'NOFRAMES'
    t_A   = r'A'
    t_COL   = r'COL'
    t_POSTER   = r'POSTER'
    t_TOP   = r'TOP'
    t_BEGIN   = r'BEGIN'
    t_TEMPORARY   = r'TEMPORARY'
    t_OFFSET   = r'OFFSET'
    t_AREA   = r'AREA'
    t_MINUS   = r'\-'
    t_LIKE   = r'LIKE'
    t_DEL   = r'DEL'
    t_INSERT   = r'INSERT'
    t_SPAN   = r'SPAN'
    t_TD   = r'TD'
    t_BDI   = r'BDI'
    t_KEYGEN   = r'KEYGEN'
    t_ASIDE   = r'ASIDE'
    t_AWK   = r'AWK'
    t_WHEN   = r'WHEN'
    t_DOCUMENT   = r'DOCUMENT\[\\COOKIE\\\]'
    t_IFRAME   = r'IFRAME'
    t_NAV   = r'NAV'
    t_OUTPUT   = r'OUTPUT'
    t_COMMAND   = r'COMMAND'
    t_PING   = r'PING'
    t_HEADER   = r'HEADER'
    t_DESC   = r'DESC'
    t_CASE   = r'CASE'
    t_IF   = r'IF'
    t_DIV   = r'DIV'
    t_DROP   = r'DROP'
    t_FROM   = r'FROM'
    t_LS   = r'LS'
    t_CREATE   = r'CREATE'
    t_LPAREN   = r'\('
    t_FONT   = r'FONT'
    t_SLEEP   = r'SLEEP'
    t_Q   = r'Q'
    t_DATALIST   = r'DATALIST'
    t_VIDEO   = r'VIDEO'
    t_MAP   = r'MAP'
    t_SUBSTRING   = r'SUBSTRING'
    t_DT   = r'DT'
    t_split   = r'[ \t\r\n\0]'
    t_WITH   = r'WITH'
    t_MENU   = r'MENU'
    t_ISINDEX   = r'ISINDEX'
    t_SMALL   = r'SMALL'
    t_LINK   = r'LINK'
    t_P   = r'P'
    t_BASEFONT   = r'BASEFONT'
    t_EMBED   = r'EMBED'
    t_PARAM   = r'PARAM'
    t_FORMATION   = r'FORMATION'
    t_FRAME   = r'FRAME'
    t_DECLARE   = r'DECLARE'
    t_BLOCKQUOTE   = r'BLOCKQUOTE'
    t_SCRIPT   = r'SCRIPT'
    t_ROLLUP   = r'ROLLUP'
    t_HASH   = r'[a-h0-9]+'
    t_SRC   = r'SRC'
    t_MSCRIPT   = r'MSCRIPT'
    t_U   = r'U'
    t_BDO   = r'BDO'
    t_MODE   = r'MODE'
    t_SELECT   = r'SELECT'
    t_GOTO   = r'goto'
    t_WAITFOR   = r'WAITFOR'
    t_META   = r'META'
    t_CAT   = r'CAT'
    t_PROCEDURE   = r'PROCEDURE'
    t_ASC   = r'ASC'
    t_COMMA   = r'\,'
    t_TILDE   = r'\~'
    t_SECTION   = r'SECTION'
    t_CURL   = r'CURL'
    t_CENTER   = r'CENTER'
    t_EQUCOMPARISON   = r'EQUCOMPARISON'
    t_SIGN   = r'GLOB'
    t_IN   = r'IN'
    t_REPLACE   = r'REPLACE'
    t_PYTHON   = r'PYTHON'
    t_CODE   = r'CODE'
    t_MAKE   = r'MAKE'
    t_FIGURE   = r'FIGURE'
    t_LEFT   = r'LEFT'
    t_SUB   = r'SUB'
    t_ARTICLE   = r'ARTICLE'
    t_IS   = r'IS'
    t_STRIKE   = r'STRIKE'
    t_MINUSMINUS   = r'\-\-'
    t_DOT   = r'\.'
    t_DD   = r'DD'
    t_SOURCE   = r'SOURCE'
    t_CANVAS   = r'CANVAS'
    t_FRAMESET   = r'FRAMESET'
    t_FOOTER   = r'FOOTER'
    t_XMLNS   = r'XMLNS\:'
    t_ELSE   = r'ELSE'
    t_DATABASE   = r'DATABASE'
    t_HAVING   = r'HAVING'
    t_EXCLA   = r'\!'
    t_DELETE   = r'DELETE'
    t_INTERVAL   = r'INTERVAL'
    t_SOME   = r'SOME'
    t_IMG   = r'IMG'
    t_VBSCRIPT   = r'VBSCRIPT'
    t_RT   = r'RT'
    t_CASCADE   = r'CASCADE'
    t_FILTER   = r'FILTER'
    t_BENCHMARK   = r'BENCHMARK'
    t_WORD   = r'[a-z0-9]+'
    #t_keyword_end   = r'#keyworld'
    t_DISTINCT   = r'DISTINCT'
    t_DFN   = r'DFN'
    t_PING6   = r'PING6'
    t_INNER   = r'INNER'
    t_AS   = r'AS'
    t_CHAR   = r'CHAR'
    t_VAR   = r'VAR'
    t_IP   = r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'
    t_SEMI   = r'\;'
    t_PLUS   = r'\+'
    t_PS   = r'PS'
    t_INS   = r'INS'
    t_PASSWORD   = r'PASSWORD'
    #t_keyword_start   = r'#keyworld'
    t_BINARY   = r'BINARY'
    t_REGEXP   = r'REGEXP'
    t_IGNORE   = r'IGNORE'
    t_XLINK   = r'XLINK\:'
    t_DUMPFILE   = r'DUMPFILE'
    t_LEGEND   = r'LEGEND'
    t_PRE   = r'PRE'
    t_SQS   = r'\`'
    t_END   = r'END'
    t_JSCRIPT   = r'JSCRIPT'
    t_ACTION   = r'ACTION'
    t_BETWEEN   = r'BETWEEN'
    t_BY   = r'BY'
    t_DOCTYPE   = r'DOCTYPE'
    t_ANY   = r'ANY'
    t_OBJECT   = r'OBJECT'
    t_BIN   = r'BIN'
    t_DELAY   = r'DELAY'
    t_TR   = r'TR'
    t_FIGCAPTION   = r'FIGCAPTION'
    t_QUICK   = r'QUICK'
    t_PROMPT   = r'PROMPT'
    t_SET   = r'SET'
    t_SAMP   = r'SAMP'
    t_UPDATE   = r'UPDATE'
    t_CITE   = r'CITE'
    t_HTML   = r'HTML'
    t_WGET   = r'WGET'
    t_LABEL   = r'LABEL'
    t_INPUT   = r'INPUT'
    t_HGROUP   = r'HGROUP'
    t_XMP   = r'XMP'
    t_STYLE   = r'STYLE'
    t_OUTER   = r'OUTER'
    t_S   = r'S'
    t_ONEVENT   = r'ON[a-z]+'
    t_SUP   = r'SUP'
    t_LI   = r'LI'
    t_KEY   = r'KEY'
    t_VALUES   = r'VALUES'
    t_TIME   = r'TIME'
    t_LBRACKET   = r'\['
    t_CAPTION   = r'CAPTION'
    t_HEAD   = r'HEAD'
    t_ATTRIBUTENAME   = r'ATTRIBUTENAME'
    t_RESTRICT   = r'RESTRICT'
    t_EXISTS   = r'EXISTS'
    t_SIGN2   = r'[\|&\^%]'
    t_METER   = r'METER'
    t_HR   = r'HR'
    t_NOT   = r'NOT'
    t_RLIKE   = r'RLIKE'
    t_ALL   = r'ALL'
    t_TRACK   = r'TRACK'
    t_RBRACKET   = r'\]'
    t_INTO   = r'INTO'
    t_MARK   = r'MARK'
    t_GROUP   = r'GROUP'
    t_NUM   = r'[0-9]+'
    t_ALERT   = r'ALERT'
    t_LIMIT   = r'LIMIT'
    t_TEXTAREA   = r'TEXTAREA'
    t_DIR   = r'DIR'
    t_ABBR   = r'ABBR'
    t_TFOOT   = r'TFOOT'
    t_B   = r'B'
    t_NOSCRIPT   = r'NOSCRIPT'
    t_H1   = r'H1'
    t_UNION   = r'UNION'
    t_CONFIRM   = r'CONFIRM'
    t_THEAD   = r'THEAD'
    t_DETAILS   = r'DETAILS'
    t_RPAREN   = r'\)'
    t_BUTTON   = r'BUTTON'
    t_ON   = r'ON'
    t_SUMMARY   = r'SUMMARY'
    t_DUPLICATE   = r'DUPLICATE'
    t_PASSWD   = r'PASSWD'
    t_PARTITION   = r'PARTITION'
    t_TITLE   = r'TITLE'


    # A regular expression rule with some action code
    #def t_NUMBER(t):
    #    r'\d+'
    #    t.value = int(t.value)
    #    return t

    # Define a rule so we can track line numbers
    #def t_newline(t):
    #    r'\n+'
    #    t.lexer.lineno += len(t.value)

    # A string containing ignored characters (spaces and tabs)
    #t_ignore  = ' \t'

    # Error handling rule
    def t_error(t):
        #print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer
    lexer = lex.lex()
    res=[]
    for data in data:
     # Give the lexer some input
        lexer.input(data)
     # Tokenize
        re=""
        while True:
            tok = lexer.token()
            if not tok:
                break      # No more input
            if tok.type == 'split':
                continue
            re=tok.type+" "+re
        res.append(re)
    return res
