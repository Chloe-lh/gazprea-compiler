#ifndef GAZPREABASE_INCLUDE_COMPILETIMEEXCEPTIONS_H_
#define GAZPREABASE_INCLUDE_COMPILETIMEEXCEPTIONS_H_

#include <string>
#include <sstream>
#include <iostream>

extern bool VERBOSE_ERRORS;

class CompileTimeException : public std::exception {
protected:
    std::string msg;

public:
    const char *what() const noexcept override {
        return msg.c_str();
    }
};

#define DEF_COMPILE_TIME_EXCEPTION(NAME)                                         \
class NAME : public CompileTimeException {                                       \
public:                                                                          \
    NAME(unsigned line, const std::string &description) {                        \
        std::stringstream buf;                                                   \
        if(VERBOSE_ERRORS){                                                      \
            buf << #NAME << " on line " << line << ": " << description << std::endl; \
        } else {                                                                 \
            buf << #NAME << " on line " << line << std::endl;                    \
        }                                                                        \
        msg = buf.str();                                                         \
    }                                                                            \
}

DEF_COMPILE_TIME_EXCEPTION(AliasingError);

DEF_COMPILE_TIME_EXCEPTION(AssignError);

DEF_COMPILE_TIME_EXCEPTION(CallError);

DEF_COMPILE_TIME_EXCEPTION(DefinitionError);

DEF_COMPILE_TIME_EXCEPTION(GlobalError);

DEF_COMPILE_TIME_EXCEPTION(MainError);

DEF_COMPILE_TIME_EXCEPTION(ReturnError);

DEF_COMPILE_TIME_EXCEPTION(SizeError);

DEF_COMPILE_TIME_EXCEPTION(StatementError);

DEF_COMPILE_TIME_EXCEPTION(SymbolError);

DEF_COMPILE_TIME_EXCEPTION(SyntaxError);

DEF_COMPILE_TIME_EXCEPTION(LiteralError);

DEF_COMPILE_TIME_EXCEPTION(TypeError);

DEF_COMPILE_TIME_EXCEPTION(StrideError);

#endif // GAZPREABASE_INCLUDE_COMPILETIMEEXCEPTIONS_H_

