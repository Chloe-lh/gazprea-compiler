#pragma once
#include <string>
#include <vector>
#include <memory>

//abstract class that is extended by the different passes in the pipeline
class ASTVisitor;

class ASTNode{ //virtual class
    public:
        // type
        virtual ~ASTNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;
};

class ExprNode : public ASTNode {
    public:
        virtual ~ExprNode() = default;
        virtual void accept(ASTVisitor& visitor) = 0;

};

class FileNode: public ASTNode{
    public:
        explicit FileNode();
        void accept(ASTVisitor& visitor) override;
};

class IntNode : public ExprNode {
    public:
        int value;
        explicit IntNode(int v); //constructor
        void accept(ASTVisitor& visitor) override;
};

class IdNode : public ExprNode {
    public:
        const std::string id;
        explicit IdNode(const std::string& id);
        void accept(ASTVisitor& visitor) override;
};



