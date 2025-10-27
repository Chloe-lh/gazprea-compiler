#include "Scope.h"

#include <iostream>
#include <stdexcept>

Scope::Scope(Scope* parent) : parent_(parent) {}

bool Scope::declare(const std::string& identifier, ValueType type) {
    if (symbols_.find(identifier) != symbols_.end()) {
        return false;
    }

    symbols_.emplace(identifier, SymbolInfo{identifier, type});
    return true;
}

SymbolInfo* Scope::resolve(const std::string& identifier) {
    auto it = symbols_.find(identifier);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolve(identifier);
    }
    return nullptr;
}

const SymbolInfo* Scope::resolve(const std::string& identifier) const {
    auto it = symbols_.find(identifier);
    if (it != symbols_.end()) {
        return &it->second;
    }
    if (parent_ != nullptr) {
        return parent_->resolve(identifier);
    }
    return nullptr;
}

void Scope::leaveStartOfBlock() {
    this->startOfBlock = false;
}

bool Scope::isStartOfBlock() {
    return this->startOfBlock;
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
    result += ">>\n";
    return result;
} 
