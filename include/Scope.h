
#include "Types.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>


struct VarInfo {
    std::string identifier;
    CompleteType type;
    bool isConst;
};

struct FuncInfo {
    std::string identifier;
    std::vector<VarInfo> params;
    CompleteType funcReturn;
};

class Scope {
public:
    explicit Scope(Scope* parent = nullptr);

    bool declareVar(const std::string& identifier, CompleteType type, bool isConst);
    bool declareFunc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType);

    VarInfo* resolveVar(const std::string& identifier);
    const VarInfo* resolveVar(const std::string& identifier) const;
    void disableDeclarations(); // For ensuring declrs are at the top of each block
    bool isDeclarationAllowed();

    Scope* parent() const { return parent_; }
    const std::unordered_map<std::string, VarInfo>& symbols() const { return symbols_; }

    // Persistent tree support
    Scope* createChild();
    const std::vector<std::unique_ptr<Scope>>& children() const { return children_; }

    // Diagnostics
    static void printAllScopes(const Scope& root);
    std::string printScope() const;

private:
    
    std::unordered_map<std::string, VarInfo> symbols_; // variables in scope
    std::unordered_map<std::string, FuncInfo> functionsBySig_; // functions in scope keyed by identifier + params, e.g. name(t1,t2,...)

    // Build canonical function key preserving parameter order
    static std::string makeFunctionKey(const std::string& identifier, const std::vector<VarInfo>& params);

    Scope* parent_;
    std::vector<std::unique_ptr<Scope>> children_;
    bool declarationAllowed = true;
};
