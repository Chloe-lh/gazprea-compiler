#include "Scope.h"

#include <iostream>
#include <stdexcept>
#include <string>

Scope::Scope(Scope* parent) : parent_(parent) {}

bool Scope::declareVar(const std::string& identifier, CompleteType type, bool isConst) {
    if (symbols_.find(identifier) != symbols_.end()) {
        return false;
    }

    symbols_.emplace(identifier, VarInfo{identifier, type, isConst});
    return true;
}

bool Scope::declareFunc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType) {
    std::string key = Scope::makeFunctionKey(identifier, params);
    if (functionsBySig_.find(key) != functionsBySig_.end()) {
        return false; // duplicate signature
    }

    FuncInfo newFunc = {
        identifier,
        params,
        returnType
    };

    functionsBySig_.emplace(std::move(key), newFunc);
    return true;
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
