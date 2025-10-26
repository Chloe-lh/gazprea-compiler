#include "CompileTimeExceptions.h"
#include <antlr4-runtime.h>

class ErrorListener : public antlr4::BaseErrorListener {
    void syntaxError(antlr4::Recognizer *recognizer, antlr4::Token * offendingSymbol,
                     size_t line, size_t charPositionInLine, const std::string &msg,
                     std::exception_ptr e) override {
        // Add rule stack for more detailed debugging
        std::vector<std::string> rule_stack = ((antlr4::Parser*) recognizer)->getRuleInvocationStack();
        std::reverse(rule_stack.begin(), rule_stack.end());
        std::ostringstream oss;
        oss << "Syntax error at line " << line << ":" << charPositionInLine << "\n"
            << "Message: " << msg << "\n"
            << "Rule stack: ";
        for (auto &rule : rule_stack) {
            oss << rule << " ";
        }

        throw SyntaxError(line, oss.str());
    }
};