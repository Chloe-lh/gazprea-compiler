#pragma once
#include "AST.h"
#include "GazpreaBaseVisitor.h"

class ASTVisitor{
    public:
        virtual void visit(FileNode* node) = 0;
        virtual void visit(IntNode* node) = 0;
        virtual void visit(IdNode* node) = 0;
    

};