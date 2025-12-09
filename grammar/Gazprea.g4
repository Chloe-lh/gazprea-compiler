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

// added support for arrays in ExplicitTypedDec -> checks legality in semantic analysis
// ei const Integer[][] id;
dec
    : qualifier? (builtin_type ID | ID size? ID) (EQ expr)? END   #ExplicitTypedDec
    | qualifier ID EQ expr END                              #InferredTypeDec
    | qualifier? tuple_dec ID (EQ expr)? END                #TupleTypedDec
    | qualifier? struct_dec (ID (EQ expr)?)? END              #StructTypedDec
    ;

stat
    : ID (COMMA ID)+ EQ expr END                #DestructAssignStat
    | tuple_access EQ expr END                  #TupleAccessAssignStat 
    | array_access EQ expr END                  #ArrayAccessAssignStat    
    | (struct_access|array_access) EQ expr END   #StructAccessAssignStat
    | { this->_input->LA(2) == GazpreaParser::EQ }? ID EQ expr END   #AssignStat
    | expr '->' STD_OUTPUT END                 #OutputStat
    | ID '<-' STD_INPUT  END                                  #InputStat
    | BREAK END                                               #BreakStat
    | CONTINUE END                                            #ContinueStat
    | RETURN expr? END                                        #ReturnStat
    | CALL ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END  #CallStat
    | struct_access PARENLEFT (expr (COMMA expr)*)? PARENRIGHT END #MethodCallStat
    | if_stat                             #IfStat
    | loop_stat                           #LoopStat
    ;

type 
    : builtin_type
    | tuple_dec
    | struct_dec
    | ID size?                        // type aliasing
    ;

// Built-in scalar types (used to disambiguate declarations)
builtin_type
    : BOOLEAN size?
    | CHARACTER size?
    | INTEGER size?
    | REAL size?
    | VECTOR '<' type '>'  
    | STRING
    ;

builtin_func
    : LENGTH PARENLEFT ID PARENRIGHT
    | SHAPE PARENLEFT ID PARENRIGHT
    | REVERSE PARENLEFT ID PARENRIGHT
    | FORMAT PARENLEFT expr PARENRIGHT
    ;

// size specification for an array
size
  : SQLEFT (INT | MULT) SQRIGHT (SQLEFT (INT|MULT) SQRIGHT)? // only up to 2D
  ;

type_alias
    : TYPEALIAS tuple_dec ID END  #TupleTypeAlias
    | TYPEALIAS struct_dec ID END #StructTypeAlias
    | TYPEALIAS type ID END   #BasicTypeAlias
    ;

expr
    : tuple_access                                      #TupleAccessExpr 
    | struct_access                                     #StructAccessExpr  
    | array_access                                      #ArrayAccessExpr
    | ID SQLEFT rangeExpr SQRIGHT                       #ArraySliceExpr
    | SQLEFT generatorBody SQRIGHT                      #GeneratorExpr
    | ID BY expr                                        #ArrayStrideExpr
    | ID PARENLEFT (expr (COMMA expr)*)? PARENRIGHT     #FuncCallExpr // Also used as struct_literal
    | struct_access PARENLEFT (expr (COMMA expr)*)? PARENRIGHT #MethodCallExpr
    | builtin_func                                      #BuiltInFuncExpr
    | PARENLEFT expr PARENRIGHT                         #ParenExpr
    | STRING_LIT                                        #StringExpr
    | <assoc=right>NOT expr                             #NotExpr
    | <assoc=right> (ADD|MINUS) expr                    #UnaryExpr
    | <assoc=right> expr EXP expr                       #ExpExpr
    | expr DOTPROD expr                                 #DotExpr
    | expr op=(MULT|DIV|REM) expr                       #MultExpr
    | expr op=(ADD|MINUS) expr                          #AddExpr
    | expr op=(LT|GT|LTE|GTE) expr                      #CompExpr
    | expr op=(EQEQ|NE) expr                            #EqExpr
    | expr AND expr                                     #AndExpr
    | expr op=(OR|XOR) expr                             #OrExpr
    | <assoc=right> expr CONCAT expr                    #ConcatExpr
    | TRUE                                              #TrueExpr
    | FALSE                                             #FalseExpr
    | CHAR                                              #CharExpr
    | INT                                               #IntExpr
    | real                                              #RealExpr
    | tuple_literal                                     #TupleLitExpr
    | array_literal                                     #ArrayLitExpr
    | AS '<' type '>' PARENLEFT expr PARENRIGHT         #TypeCastExpr
    | AS '<' tuple_dec  '>' PARENLEFT expr PARENRIGHT   #TupleTypeCastExpr
    | STD_INPUT                                         #StdInputExpr
    | ID                                                #IdExpr
    ;

generatorBody
    : generatorDomains '|' expr
    ;

generatorDomains
    : generatorDomain (COMMA generatorDomain)?
    ;

generatorDomain
    : ID IN (rangeExpr | array_literal | expr)
    ;


// Tuples
tuple_dec: TUPLE PARENLEFT type (COMMA type)+ PARENRIGHT;
tuple_literal: PARENLEFT expr (COMMA expr)+ PARENRIGHT;
tuple_access: ID DECIM INT
            | TUPACCESS
            ;

// Structs
struct_dec: STRUCT ID PARENLEFT (type ID (COMMA type ID)*)? PARENRIGHT;
struct_literal: ID PARENLEFT expr (COMMA expr)* PARENRIGHT;
struct_access: ID '.' ID;

// Arrays
array_literal : SQLEFT exprList? SQRIGHT;
array_access :  ID SQLEFT expr SQRIGHT (SQLEFT expr SQRIGHT)?; // Support expressions as indices and 2D arrays

exprList : expr (COMMA expr)* ;
rangeExpr : RANGE expr (BY expr)?          // ..end [by s]
          | expr RANGE (BY expr)?          // start.. [by s]
          | expr RANGE expr (BY expr)?     // start..end [by s]
          ;

// Block: declarations allowed anywhere but semantic analysis enforces that they appear before statements within each block.
block: CURLLEFT (dec | stat)* CURLRIGHT;

if_stat: IF PARENLEFT expr PARENRIGHT (block|stat|dec) (ELSE (block|stat|dec))?;

loop_stat
    : LOOP (block|stat) (WHILE PARENLEFT expr PARENRIGHT END)? #LoopDefault
    | LOOP (WHILE PARENLEFT expr PARENRIGHT) (block|stat) #WhileLoopBlock
    | LOOP ID IN (rangeExpr | expr) (block|stat) #ForLoopBlock
    ;

qualifier: VAR //mutable
        | CONST //immutable -  DEFAULT
        ; //annotate AST with mutability flag

// Real literals are recognised entirely at the lexer level via FLOAT.
// This parser rule is just a wrapper so we can attach AST nodes.
real
    : FLOAT
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

// Floating-point literals, including optional exponent parts.
// Examples: .0, 0.5, 32., 1e10, 1.2e-3, .5E+2
FLOAT
    : INT? DECIM INT ([eE] [+-]? INT)?        // .0, .1, .1e10, 32.0
    | INT DECIM {_input->LA(1) != '.'}? ([eE] [+-]? INT)? // 32., 32.e+3 (guarded against ..)
    | INT [eE] [+-]? INT                      // 1e10, 1e-3
    ;


// operators and punctuation
END: ';';

ADD: '+';
CONCAT: '||';
MINUS: '-';
DOTPROD: '**';
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
SHAPE: 'shape';
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
