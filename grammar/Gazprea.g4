grammar Gazprea;

file: (dec|func|procedure|type_alias|stat)* EOF;

func
    : FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS type block       #FunctionBlock
    | FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS tuple_dec block  #FunctionBlockTupleReturn
    | FUNCTION ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT RETURNS type EQ expr END #FunctionStat
    | FUNCTION ID PARENLEFT (type ID? (COMMA type ID?)*)? PARENRIGHT RETURNS type END       #FunctionPrototype
    | FUNCTION ID PARENLEFT (type ID? (COMMA type ID?)*)? PARENRIGHT RETURNS tuple_dec END  #FunctionPrototypeTupleReturn
    ;

procedure
    : PROCEDURE ID PARENLEFT (param (COMMA param)*)? PARENRIGHT (RETURNS type)? block       #ProcedureBlock
    | PROCEDURE ID PARENLEFT (param (COMMA param)*)? PARENRIGHT (RETURNS type)? END     #ProcedurePrototype
    ;

param: qualifier? type ID;

dec
    : qualifier? (builtin_type ID | ID ID) (EQ expr)? END   #ExplicitTypedDec
    | qualifier ID EQ expr END                              #InferredTypeDec
    | qualifier? tuple_dec ID (EQ expr)? END                #TupleTypedDec
    ;

stat
    : ID (COMMA ID)+ EQ expr END                #DestructAssignStat
    | tuple_access EQ expr END                  #TupleAccessAssignStat     
    | tuple_access '->' STD_OUTPUT END          #OutputStat
    | { this->_input->LA(2) == GazpreaParser::EQ }? ID EQ expr END   #AssignStat
    | expr '->' STD_OUTPUT END      #OutputStat
    | ID '<-' STD_INPUT  END        #InputStat
    | BREAK END                     #BreakStat
    | CONTINUE END                  #ContinueStat
    | RETURN expr? END              #ReturnStat
    | CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
    | if_stat                             #IfStat
    | loop_stat                           #LoopStat
    ;

type //this should include basic types
    : BOOLEAN
    | CHARACTER
    | INTEGER
    | REAL
    | STRING
    | ID
    ;

// Built-in scalar types (used to disambiguate declarations)
builtin_type
    : BOOLEAN
    | CHARACTER
    | INTEGER
    | REAL
    | STRING
    ;

type_alias
    : TYPEALIAS type ID END   #BasicTypeAlias
    | TYPEALIAS tuple_dec ID END  #TupleTypeAlias
    ;


expr
    : tuple_access                                      #TupleAccessExpr
    | ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT     #FuncCallExpr
    | PARENLEFT expr PARENRIGHT                         #ParenExpr
    | STRING_LIT                                       #StringExpr
    | <assoc=right>NOT expr                             #NotExpr
    | <assoc=right> (ADD|MINUS) expr                    #UnaryExpr
    | <assoc=right> expr EXP expr                       #ExpExpr
    | expr op=(MULT|DIV|REM) expr                       #MultExpr
    | expr op=(ADD|MINUS) expr                          #AddExpr
    | expr op=(LT|GT|LTE|GTE) expr                      #CompExpr
    | expr op=(EQEQ|NE) expr                            #EqExpr
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
tuple_access: ID DECIM INT
            | TUPACCESS
            ;

// declarations must be placed at the start of the block
block: CURLLEFT dec* stat* CURLRIGHT;

if_stat: IF PARENLEFT expr PARENRIGHT (block|stat|dec) (ELSE (block|stat|dec))?;

loop_stat
    : LOOP (block|stat) (WHILE PARENLEFT expr PARENRIGHT END)? #LoopDefault
    | LOOP (WHILE PARENLEFT expr PARENRIGHT) (block|stat) #WhileLoopBlock
    | LOOP ID IN (rangeExpr | arrayLiteral) (block|stat) #ForLoopBlock
    ;

rangeExpr: expr RANGE expr;

arrayLiteral: SQLEFT expr (COMMA expr)* SQRIGHT;


qualifier: VAR //mutable
        | CONST //immutable -  DEFAULT
        ; //annotate AST with mutability flag

real
    : FLOAT (UPPER_E|LOWER_E)? (ADD INT | MINUS INT | INT)?
    | INT (UPPER_E|LOWER_E) (ADD|MINUS)? INT
    ;

CHAR: '\'' (ESC_SEQ | ~[\\']) '\'';
STRING_LIT: '"' (ESC_SEQ | ~('\\'|'"'))* '"';

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
EQEQ: '==';
EQ: '=';

// for floating point
DECIM: '.';
UPPER_E: 'E';
LOWER_E: 'e';

COMMA: ',';

// Recognize tuple member access as a single token to avoid ambiguity
TUPACCESS: [a-zA-Z_][a-zA-Z0-9_]* '.' [0-9]+;

// brackets
CURLLEFT: '{';
CURLRIGHT: '}';
PARENLEFT: '(';
PARENRIGHT: ')';
SQLEFT: '[';
SQRIGHT: ']';
RANGE: '..';

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

// Place after TUPACCESS so 'tup.1' is not split into ID '.' INT
ID: [a-zA-Z_][a-zA-Z0-9_]*;

//skip whitespace and comments
SL_COMMENT: '//'.*? ('\n'|EOF) -> skip; 
ML_COMMENT: '/*' .*? '*/' -> skip; //cannot be nested
WS : [ \t\r\n]+ -> skip ;
