#pragma once
#include "CompileTimeExceptions.h"
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

struct ProcInfo {
    std::string identifier;
    std::vector<VarInfo> params;
    CompleteType procReturn;
};

class Scope {
public:
    explicit Scope(Scope* parent = nullptr);
    explicit Scope(Scope* parent = nullptr, bool inLoop, const CompleteType* returnType);

    void declareVar(const std::string& identifier, const CompleteType& type, bool isConst);
    void declareFunc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType);
    void declareProc(const std::string& identifier, const std::vector<VarInfo>& params, const CompleteType& returnType);
    void declareAlias(const std::string& identifier, const CompleteType& type);

    VarInfo* resolveVar(const std::string& identifier);
    FuncInfo* resolveFunc(const std::string& identifier, const std::vector<VarInfo>& params);
    ProcInfo* resolveProc(const std::string& identifier, const std::vector<VarInfo>& params);
    CompleteType* resolveAlias(const std::string& identifier);

    void disableDeclarations(); // For ensuring declrs are at the top of each block
    bool isDeclarationAllowed();
    void setGlobalTrue();

    bool isInLoop();
    bool isInFunction();
    const CompleteType* getReturnType();

    Scope* parent() const { return parent_; }
    const std::unordered_map<std::string, VarInfo>& symbols() const { return symbols_; }

    // Persistent tree support
    Scope* createChild(const bool inLoop, const CompleteType* returnType);
    const std::vector<std::unique_ptr<Scope>>& children() const { return children_; }

    // Diagnostics
    static void printAllScopes(const Scope& root);
    std::string printScope() const;

private:
    static std::unordered_map<std::string, CompleteType> globalTypeAliases_; // type aliases can only be declared in global scope
    
    std::unordered_map<std::string, VarInfo> symbols_; // variables in scope
    // Functions, procedures and (pt2) structs will share same namespace
    std::unordered_map<std::string, FuncInfo> functionsByName_; // functions in scope keyed by identifier
    std::unordered_map<std::string, ProcInfo> proceduresByName_; // procedures in scope keyed by identifier


    Scope* parent_;
    std::vector<std::unique_ptr<Scope>> children_;
    bool declarationAllowed = true;
    bool isGlobal = false;
    bool inLoop = false;
    const CompleteType* returnType = nullptr; // nullptr if not inside a function
};
