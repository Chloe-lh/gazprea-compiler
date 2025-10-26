grammar Gazprea;

file: .*? EOF;

func
    : FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS type block       #FunctionBlock
    | FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS tuple_dec block  #FunctionBlockTupleReturn
    | FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS type EQ stat     #FunctionStat
    | FUNCTION ID PARENLEFT (type ID? (COMMA type ID?)*)? PARENRIGHT RETURNS type END       #FunctionPrototype
    | FUNCTION ID PARENLEFT (type ID? (COMMA type ID?)*)? PARENRIGHT RETURNS tuple_dec END  #FunctionPrototypeTupleReturn
    ;

procedure: PROCEDURE ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT block;

dec
    : qualifier? type ID (EQ expr)? END          #ExplicitTypedDec
    | qualifier ID EQ expr END                   #InferredTypeDec
    | qualifier? tuple_dec ID (EQ expr)? END     #TupleTypedDec
    ;

stat
    : ID EQ expr END                #AssignStat
    | expr '->' STD_OUTPUT END      #OutputStat
    | ID '<-' STD_INPUT  END        #InputStat
    | BREAK END                     #BreakStat
    | CONTINUE END                  #ContinueStat
    | RETURN expr? END              #ReturnStat
    | CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
    ;

type //this should include basic types
    : BOOLEAN
    | CHARACTER
    | INTEGER
    | REAL
    | ID
    ;

type_alias
    : TYPEALIAS ID ID   #BasicTypeAlias
    | TYPEALIAS tuple_dec ID  #TupleTypeAlias
    ;


expr
    : tuple_access                                      #TupleAccessExpr
    | ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT     #FuncCallExpr
    | PARENLEFT expr PARENRIGHT                         #ParenExpr
    | <assoc=right> (ADD|MINUS) expr                    #UnaryExpr
    | <assoc=right> expr EXP expr                       #ExpExpr
    | expr op=(MULT|DIV|REM) expr                       #MultExpr
    | expr op=(ADD|MINUS) expr                          #AddExpr
    | expr op=(LT|GT|LTE|GTE) expr                      #CompExpr
    | <assoc=right>NOT expr                             #NotExpr
    | expr op=(EQ|NE) expr                              #EqExpr
    | expr AND expr                                     #AndExpr
    | expr op=(OR|XOR) expr                             #OrExpr
    | TRUE                                              #TrueExpr
    | FALSE                                             #FalseExpr
    | CHAR                                              #CharExpr
    | INT                                               #IntExpr
    | real                                              #RealExpr
    | tuple_literal                                     #TupleLitExpr
    | AS '<' type '>' PARENLEFT expr PARENRIGHT         #TypeCastExpr
    | AS '<' tuple_dec  '>' PARENLEFT expr PARENRIGHT   #TupleTypeCastExpr
    | ID                                                #IdExpr
    ;

tuple_dec: TUPLE PARENLEFT type (COMMA type)+ PARENRIGHT;
tuple_literal: PARENLEFT expr (COMMA expr)+ PARENRIGHT;
tuple_access: ID DECIM TUPLE_INT;
TUPLE_INT: [1-9][0-9]*;

// declarations must be placed at the start of the block
block: CURLLEFT dec* stat* CURLRIGHT;

if: IF PARENLEFT expr PARENRIGHT (block|stat) (ELSE (block|stat))?;

loop
    : LOOP (block|stat) (WHILE PARENLEFT expr PARENRIGHT END)? #Loop
    | LOOP (WHILE PARENLEFT expr PARENRIGHT) (block|stat) #WhileLoopBlock
    ;


qualifier: VAR //mutable
        | CONST //immutable -  DEFAULT
        ; //annotate AST with mutability flag

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
STRUCT: 'struct';

//skip whitespace and comments
SL_COMMENT: '//'.*? -> skip; 
ML_COMMENT: '/*' .*? '*/' -> skip; //cannot be nested
WS : [ \t\r\n]+ -> skip ;



