grammar Gazprea;

file: .*? EOF;


dec
    : qualifier? type ID (EQ expr)?                                #ExplicitTypedDec //it needs to be type|ID to account for aliases
    | qualifier ID EQ expr                                         #InferredTypeDec
    | qualifier? TUPLE PARENLEFT type (COMMA type)+ PARENRIGHT ID (EQ expr)?  #TupleTypedDec 
    ;

stat
    : ID EQ expr
    ;

type //this should include basic types
    : BOOLEAN
    | CHARACTER
    | INTEGER
    | REAL
    | ID
    ;

expr
    : PARENLEFT expr PARENRIGHT          #ParenExpr
    | <assoc=right> (ADD|MINUS) expr     #UnaryExpr
    | <assoc=right> expr EXP expr        #ExpExpr
    | expr op=(MULT|DIV|REM) expr        #MultExpr
    | expr op=(ADD|MINUS) expr           #AddExpr
    | expr op=(LT|GT|LTE|GTE) expr       #CompExpr
    | <assoc=right>NOT expr              #NotExpr
    | expr op=(EQ|NE) expr               #EqExpr
    | expr AND expr                      #AndExpr
    | expr op=(OR|XOR) expr              #OrExpr
    | TRUE                               #TrueExpr
    | FALSE                              #FalseExpr
    | CHAR                               #CharExpr
    | INT                                #IntExpr
    | real                               #RealExpr
    | tuple_literal                      #TupleLitExpr
    | tuple_access                       #TupleAccessExpr
    | ID                                 #IdExpr
    ;

tuple_literal: PARENLEFT expr (COMMA expr)+ PARENRIGHT;
tuple_access: ID DECIM TUPLE_INT;


// declarations must be placed at the start of the block
block: CURLLEFT dec* stat* CURLRIGHT;
qualifier: VAR //mutable
        | CONST //immutable -  DEFAULT
        ; //annotate AST with mutability flag

// Expression precedence:
// paren > index > range > mult/div >add/sub > rem/exp > un neg/plus > lt/gt > eq/neq

real
    : FLOAT (UPPER_E|LOWER_E)? (ADD INT | MINUS INT | INT)?
    | INT (UPPER_E|LOWER_E) (ADD|MINUS)? INT
    ;

CHAR: '\'' (ESC_SEQ | ~[\\']) '\'';

fragment ESC_SEQ:
      '\\0'  // Null
    | '\\a'  // Bell
    | '\\b'  // Backspace
    | '\\t'  // Tab
    | '\\n'  // Line Feed
    | '\\r'  // Carriage Return
    | '\\"'  // Quotation Mark
    | '\\\'' // Apostrophe
    | '\\\\' // Backslash
    ;
INT: [0-9]+;
TUPLE_INT: [1-9][0-9]+;

FLOAT
    : INT? DECIM INT // .0
    | INT DECIM INT?; // 32.

ID: [a-zA-Z_][a-zA-Z0-9_]*;

// operators and punctuation
END: ';';

ADD: '+';
MINUS: '-';
MULT: '*';
DIV: '/';
REM: '%';
EXP: '^';
LT: '<';
GT: '>';
LTE: '<=';
GTE: '>=';
NE:'!=';
EQ: '=';

// for floating point
DECIM: '.';
UPPER_E: 'E';
LOWER_E: 'e';

COMMA: ',';

// brackets
CURLLEFT: '{';
CURLRIGHT: '}';
PARENLEFT: '(';
PARENRIGHT: ')';
SQLEFT: '[';
SQRIGHT: ']';

// keywords
AND: 'and';
AS: 'as';
BOOLEAN: 'boolean';
BREAK: 'break';
BY: 'by';
CALL: 'call';
CHARACTER: 'character';
COLUMNS: 'columns';
CONST: 'const';
CONTINUE: 'continue';
ELSE: 'else';
FALSE: 'false';
FORMAT: 'format';
FUNCTION: 'function';
IF: 'if';
IN: 'in';
INTEGER: 'integer';
LENGTH: 'length';
LOOP: 'loop';
NOT: 'not';
OR: 'or';
PROCEDURE: 'procedure';
REAL: 'real';
RETURN: 'return';
RETURNS: 'returns';
REVERSE: 'reverse';
ROWS: 'rows';
STD_INPUT: 'std_input';
STD_OUTPUT: 'std_output';
STREAM_STATE: 'stream_state';
STRING: 'string';
TRUE: 'true';
TUPLE: 'tuple';
TYPEALIAS: 'typealias';
VAR: 'var';
VECTOR: 'vector';
WHILE: 'while';
XOR: 'xor';

//skip whitespace and comments
SL_COMMENT: '//'.*? -> skip; 
ML_COMMENT: '/*' .*? '*/' -> skip; //cannot be nested
WS : [ \t\r\n]+ -> skip ;



