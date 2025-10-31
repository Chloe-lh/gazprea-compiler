#include "Scope.h"

#include <iostream>
#include <stdexcept>
#include <string>

Scope::Scope(Scope* parent) : parent_(parent) {}

void Scope::declareVar(const std::string& identifier, const CompleteType& type, bool isConst) {
    if (symbols_.find(identifier) != symbols_.end()) {
        SymbolError err = SymbolError(1, "Semantic Analysis: Variable '" + identifier + "' cannot be redeclared.");
    }

    symbols_.emplace(identifier, VarInfo{identifier, type, isConst});
}

void Scope::declareFunc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType) {
    std::string key = Scope::makeFunctionKey(identifier, params);
    if (functionsBySig_.find(key) != functionsBySig_.end()) {
        SymbolError err = SymbolError(1, "Semantic Analysis: Function '" + identifier + "' cannot be redeclared.");
    }

    FuncInfo newFunc = {
        identifier,
        params,
        returnType
    };

    functionsBySig_.emplace(std::move(key), newFunc);
}

/*
TODO add error line number to 2 errors below
*/
void Scope::declareAlias(const std::string& identifier, const CompleteType& type) {
    if (!isGlobal) {
        GlobalError err = GlobalError(1, "Semantic Analysis: Cannot declare alias '" + identifier + "' in non-global scope."); 
        throw err;
    }

    if (globalTypeAliases_.find(identifier) != globalTypeAliases_.end()) {
        AliasingError err = AliasingError(1, "Semantic Analysis: Re-declaring existing alias '" + identifier + ".");
    }

    globalTypeAliases_.emplace(identifier, type);
}

FuncInfo* Scope::resolveFunc(const std::string& identifier, const std::vector<VarInfo>& params) {
    std::string key = Scope::makeFunctionKey(identifier, params);

    auto it = functionsBySig_.find(key);
    if (it != functionsBySig_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveFunc(identifier, params);
    }
    return nullptr;
}

VarInfo* Scope::resolveVar(const std::string& identifier) {
    auto it = symbols_.find(identifier);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveVar(identifier);
    }
    return nullptr;
}

const VarInfo* Scope::resolveVar(const std::string& identifier) const {
    auto it = symbols_.find(identifier);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolveVar(identifier);
    }
    return nullptr;
}

void Scope::disableDeclarations() {
    this->declarationAllowed = false;
}

bool Scope::isDeclarationAllowed() {
    return this->declarationAllowed;
}

void Scope::setGlobalTrue() {
    this->isGlobal = true;
}

Scope* Scope::createChild() {
    children_.push_back(std::make_unique<Scope>(this));
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
    for (const auto& entry : functionsBySig_) {
        result += entry.first + " -> " + toString(entry.second.funcReturn) + "\n";
    }
    result += ">>\n";
    return result;
} 

std::string Scope::makeFunctionKey(const std::string& identifier, const std::vector<VarInfo>& params) {
    std::string key;
    key.reserve(identifier.size() + 2 + params.size() * 8);
    key += identifier;
    key += '(';
    for (size_t i = 0; i < params.size(); ++i) {
        key += toString(params[i].type);
        if (i + 1 < params.size()) key += ", ";
    }
    key += ')';
    return key;
}
