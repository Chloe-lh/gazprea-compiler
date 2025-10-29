
#include "Types.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>


struct SymbolInfo {
    std::string identifier;
    CompleteType type;
    bool isConst;
};

class Scope {
public:
    explicit Scope(Scope* parent = nullptr);

    bool declare(const std::string& identifier, CompleteType type, bool isConst);

    SymbolInfo* resolve(const std::string& identifier);
    const SymbolInfo* resolve(const std::string& identifier) const;
    void disableDeclarations(); // For ensuring declrs are at the top of each block
    bool isDeclarationAllowed();

    Scope* parent() const { return parent_; }
    const std::unordered_map<std::string, SymbolInfo>& symbols() const { return symbols_; }

    // Persistent tree support
    Scope* createChild();
    const std::vector<std::unique_ptr<Scope>>& children() const { return children_; }

    // Diagnostics
    static void printAllScopes(const Scope& root);
    std::string printScope() const;

private:
    std::unordered_map<std::string, SymbolInfo> symbols_;
    Scope* parent_;
    std::vector<std::unique_ptr<Scope>> children_;
    bool declarationAllowed = true;
};
