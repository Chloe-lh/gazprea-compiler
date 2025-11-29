#include "Scope.h"
#include "CompileTimeExceptions.h"
#include "Types.h"

#include <iostream>
#include <stdexcept>
#include <string>


// Static storage for global type aliases
std::unordered_map<std::string, CompleteType> Scope::globalTypeAliases_{};

/* This constructor assumes not in function, not in loop. i.e. global/ global block*/
Scope::Scope(Scope* parent) : parent_(parent) {}

Scope::Scope(Scope* parent, bool inLoop, const CompleteType* returnType) : parent_(parent), inLoop(inLoop), returnType(returnType) {}

void Scope::declareVar(const std::string& identifier, const CompleteType& type, bool isConst, int line) {
    if (symbols_.find(identifier) != symbols_.end()) {
        SymbolError err = SymbolError(line, "Semantic Analysis: Variable '" + identifier + "' cannot be redeclared.");
        throw err;
    }

    symbols_.emplace(identifier, VarInfo{identifier, type, isConst});
}

void Scope::declareFunc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType, int line) {
    // Shared namespace: disallow conflict with any procedure of same name
    if (proceduresByName_.find(identifier) != proceduresByName_.end()) {
        throw SymbolError(line, "Semantic Analysis: Name conflict: function and procedure share name '" + identifier + "'.");
    }
    if (functionsByName_.find(identifier) != functionsByName_.end()) {
        SymbolError err = SymbolError(line, "Semantic Analysis: Function '" + identifier + "' cannot be redeclared.");
        throw err;
    }

    FuncInfo newFunc = { identifier, params, returnType };
    functionsByName_.emplace(identifier, newFunc);
}

void Scope::declareProc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType, int line) {
    std::cout << "declareProc Called\n";
    // Shared namespace: disallow conflict with any function of same name
    if (functionsByName_.find(identifier) != functionsByName_.end()) {
        throw SymbolError(line, "Semantic Analysis: Name conflict: function and procedure share name '" + identifier + "'.");
    }
    if (proceduresByName_.find(identifier) != proceduresByName_.end()) {
        SymbolError err = SymbolError(line, "Semantic Analysis: Procedure '" + identifier + "' cannot be redeclared.");
        throw err;
    }

    ProcInfo newProc = { identifier, params, returnType };
    proceduresByName_.emplace(identifier, newProc);
}


void Scope::declareAlias(const std::string& identifier, const CompleteType& type, int line) {
    if (!inGlobal) {
        GlobalError err = GlobalError(line, "Semantic Analysis: Cannot declare alias '" + identifier + "' in non-global scope."); 
        throw err;
    }

    if (globalTypeAliases_.find(identifier) != globalTypeAliases_.end()) {
        AliasingError err = AliasingError(line, "Semantic Analysis: Re-declaring existing alias '" + identifier + ".");
        throw err;
    }

    globalTypeAliases_.emplace(identifier, type);
}

FuncInfo* Scope::resolveFunc(const std::string& identifier, const std::vector<VarInfo>& callParams, int line) {
    auto it = functionsByName_.find(identifier);
    if (it != functionsByName_.end()) {
        // Validate parameter types (names may differ/omitted in prototypes)
        const auto& stored = it->second.params;
        if (stored.size() != callParams.size()) {
            throw SymbolError(line, "Semantic Analysis: Function '" + identifier + "' called with wrong number of arguments.");
        }
        for (size_t i = 0; i < stored.size(); ++i) {
            // Use promote logic: if paramType == promote(argType, paramType), then it's compatible
            if (promote(callParams[i].type, stored[i].type) != stored[i].type) {
                throw SymbolError(line, "Semantic Analysis: Function '" + identifier + "' called with incompatible argument types.");
            }
        }
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveFunc(identifier, callParams, line);
    }
    throw SymbolError(line, "Semantic Analysis: Function '" + identifier + "' not defined.");
}

ProcInfo* Scope::resolveProc(const std::string& identifier, const std::vector<VarInfo>& callParams, int line) {
    auto it = proceduresByName_.find(identifier);
    if (it != proceduresByName_.end()) {
        const auto& stored = it->second.params;
        if (stored.size() != callParams.size()) {
            throw SymbolError(line, "Semantic Analysis: Procedure '" + identifier + "' called with wrong number of arguments.");
        }
        for (size_t i = 0; i < stored.size(); ++i) {
             // Use promote logic
            if (promote(callParams[i].type, stored[i].type) != stored[i].type) {
                throw SymbolError(line, "Semantic Analysis: Procedure '" + identifier + "' called with incompatible argument types.");
            }
        }
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveProc(identifier, callParams, line);
    }
    throw SymbolError(line, "Semantic Analysis: Procedure '" + identifier + "' not defined.");
}

// TODO add line number in error
VarInfo* Scope::resolveVar(const std::string& identifier, int line) {
    auto it = symbols_.find(identifier);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveVar(identifier, line);
    }
    throw SymbolError(line, "Semantic Analysis: Variable '" + identifier + "' not defined.");
}

CompleteType* Scope::resolveAlias(const std::string& identifier, int line) {
    auto it = globalTypeAliases_.find(identifier);

    if (it != globalTypeAliases_.end()) {
        return &it->second;
    }

    throw SymbolError(line, "Semantic Analysis: Type alias '" + identifier + "' not defined.");
}

void Scope::disableDeclarations() {
    this->declarationAllowed = false;
}

bool Scope::isDeclarationAllowed() {
    return this->declarationAllowed;
}

void Scope::setGlobalTrue() {
    this->inGlobal = true;
}

bool Scope::isInLoop() { return this->inLoop; }
bool Scope::isInFunction() { return this->inFunction; }
bool Scope::isInGlobal() { return this->inGlobal; }
void Scope::setInFunctionTrue() { this->inFunction = true; }
const CompleteType* Scope::getReturnType() { return this->returnType; }

Scope* Scope::createChild(const bool inLoop, const CompleteType* returnType) {
    children_.push_back(std::make_unique<Scope>(this, inLoop, returnType));
    // Maintain 'inFunction' value to child scopes to prevent calls, etc. in pure functions
    children_.back()->inFunction = this->inFunction;
    return children_.back().get();
}

void Scope::printAllScopes(const Scope& root) {
    std::ostream& stream = std::cerr;
    stream << root.printScope();
    stream.flush();
}

std::string Scope::printScope() const {
    std::string result = "\n<<\n";
    for (const auto& child : children_) {
        result += child->printScope() + "\n";
    }
    for (const auto& symbol : symbols_) {
        result += symbol.second.identifier + " : " + toString(symbol.second.type) + "\n";
    }
    for (const auto& entry : functionsByName_) {
        result += entry.first + " -> " + toString(entry.second.funcReturn) + "\n";
    }
    for (const auto& entry : proceduresByName_) {
        result += entry.first + " => " + toString(entry.second.procReturn) + "\n";
    }
    result += ">>\n";
    return result;
}