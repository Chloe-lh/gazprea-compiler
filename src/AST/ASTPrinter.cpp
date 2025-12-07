#include "ASTPrinter.h"
#include "AST.h"
#include "Types.h"
#include <string>
#include <vector>

namespace gazprea {

ASTPrinter::ASTPrinter(std::ostream &output_stream, bool enableColor)
    : out(output_stream), colorEnabled(enableColor) {}

void ASTPrinter::visit(FileNode *node) {
  printTreeLine("FileNode");
  indent++;
  for (size_t i = 0; i < node->stats.size(); i++) {
    bool isLast = (i == node->stats.size() - 1);
    pushChildContext(isLast);
    if (node->stats[i]) {
      node->stats[i]->accept(*this);
    } else {
      printTreeLine("null");
    }
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(BlockNode *node) {
  printTreeLine("BlockNode");
  indent++;
  for (size_t i = 0; i < node->decs.size(); i++) {
    bool isLast = (node->stats.empty() && (i == node->decs.size() - 1));
    pushChildContext(isLast);
    node->decs[i]->accept(*this);
    popChildContext();
  }
  for (size_t i = 0; i < node->stats.size(); i++) {
    bool isLast = (i == node->stats.size() - 1);
    pushChildContext(isLast);
    node->stats[i]->accept(*this);
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(BuiltInFuncNode *node) {
  printTreeLine("BuiltInFunc", "name: " + node->funcName + ", arg: " + node->id);
}
void ASTPrinter::visit(ArrayStrideExpr *node){
  printTreeLine("ArrayStrideExpr", "id: " + node->id);
  indent++;
  pushChildContext(true);
  printTreeLine("StrideExpr");
  indent++;
  pushChildContext(true);
  if (node->expr) node->expr->accept(*this);
  else printTreeLine("<null>");
  popChildContext();
  indent--;
  popChildContext();
  indent--;
}
void ASTPrinter::visit(ArraySliceExpr *node) { 
  printTreeLine("ArraySliceExpr", "id: " + node->id);
  indent++;
  pushChildContext(true);
  printTreeLine("Range");
  indent++;
  pushChildContext(true);
  if (node->range) node->range->accept(*this);
  else printTreeLine("<null>");
  popChildContext();
  indent--;
  popChildContext();
  indent--;
}
void ASTPrinter::visit(ArrayAccessNode *node) { 
  printTreeLine("ArrayAccessNode", "id: " + node->id);
  indent++;
  pushChildContext(true);
  printTreeLine("Index");
  indent++;
  pushChildContext(true);
  if (node->indexExpr) {
    node->indexExpr->accept(*this);
  } else {
    printTreeLine("<null>");
  }
  popChildContext();
  indent--;
  popChildContext();
  indent--;
}
void ASTPrinter::visit(ArrayTypedDecNode *node) {
  if (!node) { printTreeLine("ArrayTypedDecNode", "<null>"); return; }
  printTreeLine("ArrayTypedDecNode", "name: " + node->id + ", qualifier: " + (node->qualifier.empty() ? std::string("const") : node->qualifier));
  indent++;
  // Print declared type (including any dimension info embedded in CompleteType)
  {
    bool isLast = (node->init == nullptr);
    pushChildContext(isLast);
    std::string details = "type: " + toString(node->type);
    details += " (dims:";
    for (size_t i = 0; i < node->type.dims.size(); ++i) {
      if (i) details += ",";
      int d = node->type.dims[i];
      if (d >= 0) details += std::to_string(d);
      else        details += "?";
    }
    details += ")";
    printTreeLine("Type", details);
    popChildContext();
  }

  if (node->init) {
    pushChildContext(true);
    indent++;
    pushChildContext(true);
    node->init->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  } else {
    printTreeLine("<null init>");
  }

  indent--;
}

void ASTPrinter::visit(ExprListNode *node) {
  printTreeLine("ExprListNode");
  indent++;
  for (size_t i = 0; i < node->list.size(); ++i) {
    pushChildContext(i == node->list.size() - 1);
    if (node->list[i]) node->list[i]->accept(*this);
    else printTreeLine("<null>");
    popChildContext();
  }
  indent--;
}
void ASTPrinter::visit(ArrayLiteralNode *node) {
  printTreeLine("ArrayLiteralNode");
  indent++;
  pushChildContext(true);
  if (node->list) node->list->accept(*this);
  else printTreeLine("<empty>");
  popChildContext();
  indent--;
}
void ASTPrinter::visit(RangeExprNode *node) {
  printTreeLine("RangeExprNode");
  indent++;
  if (node->start) {
    pushChildContext(node->end == nullptr);
    printTreeLine("Start");
    indent++;
    pushChildContext(true);
    node->start->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  if (node->end) {
    pushChildContext(true);
    printTreeLine("End");
    indent++;
    pushChildContext(true);
    node->end->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  if (node->step) {
    pushChildContext(true);
    printTreeLine("Step");
    indent++;
    pushChildContext(true);
    node->step->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(GeneratorExprNode *node) {
  printTreeLine("GeneratorExprNode");
  indent++;
  pushChildContext(false);
  printTreeLine("Domains");
  indent++;
  for (size_t i = 0; i < node->domains.size(); ++i) {
    pushChildContext(i + 1 < node->domains.size());
    printTreeLine("Domain", "iter: " + node->domains[i].first);
    indent++;
    pushChildContext(true);
    if (node->domains[i].second) {
      node->domains[i].second->accept(*this);
    } else {
      printTreeLine("null domain expr");
    }
    popChildContext();
    indent--;
  }
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("RHS");
  indent++;
  pushChildContext(true);
  if (node->rhs) node->rhs->accept(*this);
  else printTreeLine("null rhs");
  popChildContext();
  indent--;
  popChildContext();

  if (node->lowered) {
    pushChildContext(true);
    printTreeLine("Lowered");
    indent++;
    pushChildContext(true);
    node->lowered->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(IfNode *node) {
  printTreeLine("IfNode");
  indent++;

  pushChildContext(false);
  printTreeLine("Condition");
  indent++;
  pushChildContext(true);
  node->cond->accept(*this);
  popChildContext();
  indent--;
  popChildContext();

  pushChildContext(node->elseBlock == nullptr && node->elseStat == nullptr);
  printTreeLine("Then");
  indent++;
  pushChildContext(true);
  if (node->thenBlock)
    node->thenBlock->accept(*this);
  else
    node->thenStat->accept(*this);
  popChildContext();
  indent--;
  popChildContext();

  if (node->elseBlock || node->elseStat) {
    pushChildContext(true);
    printTreeLine("Else");
    indent++;
    pushChildContext(true);
    if (node->elseBlock)
      node->elseBlock->accept(*this);
    else
      node->elseStat->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }

  indent--;
}

void ASTPrinter::visit(LoopNode *node) {
  std::string kind;
  switch (node->kind) {
  case LoopKind::Plain:
    kind = "Plain";
    break;
  case LoopKind::While:
    kind = "While";
    break;
  case LoopKind::WhilePost:
    kind = "WhilePost";
    break;
  }
  printTreeLine("LoopNode", "kind: " + kind);
  indent++;
  if (node->cond) {
    pushChildContext(false);
    printTreeLine("Condition");
    indent++;
    pushChildContext(true);
    node->cond->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  pushChildContext(true);
  printTreeLine("Body");
  indent++;
  pushChildContext(true);
  if (node->body) {
    node->body->accept(*this);
  } else {
    printTreeLine("null body");
  }
  popChildContext();
  indent--;
  popChildContext();
  indent--;
}

void ASTPrinter::visit(IteratorLoopNode *node) {
  printTreeLine("IteratorLoopNode", "iter: " + node->iterName);
  indent++;
  pushChildContext(false);
  printTreeLine("Domain");
  indent++;
  pushChildContext(true);
  if (node->domainExpr) {
    node->domainExpr->accept(*this);
  } else {
    printTreeLine("null domain");
  }
  popChildContext();
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("Body");
  indent++;
  pushChildContext(true);
  if (node->body) {
    node->body->accept(*this);
  } else {
    printTreeLine("null body");
  }
  popChildContext();
  indent--;
  popChildContext();

  if (node->lowered) {
    pushChildContext(true);
    printTreeLine("Lowered");
    indent++;
    pushChildContext(true);
    node->lowered->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(ParenExpr *node) {
  printTreeLine("ParenExpr");
  indent++;
  pushChildContext(true);
  node->expr->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(UnaryExpr *node) {
  printTreeLine("UnaryExpr", "op: " + node->op);
  indent++;
  pushChildContext(true);
  node->operand->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(ExpExpr *node) {
  printTreeLine("ExpExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(MultExpr *node) {
  printTreeLine("MultExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(DotExpr *node) {
  printTreeLine("DotExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  if (node->left) node->left->accept(*this);
  else printTreeLine("<null left>");
  popChildContext();
  pushChildContext(true);
  if (node->right) node->right->accept(*this);
  else printTreeLine("<null right>");
  popChildContext();
  indent--;
}

void ASTPrinter::visit(AddExpr *node) {
  printTreeLine("AddExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(CompExpr *node) {
  printTreeLine("CompExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(NotExpr *node) {
  printTreeLine("NotExpr", "op: " + node->op);
  indent++;
  pushChildContext(true);
  node->operand->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(EqExpr *node) {
  printTreeLine("EqExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(AndExpr *node) {
  printTreeLine("AndExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(OrExpr *node) {
  printTreeLine("OrExpr", "op: " + node->op);
  indent++;
  pushChildContext(false);
  node->left->accept(*this);
  popChildContext();
  pushChildContext(true);
  node->right->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(TrueNode *node) { printTreeLine("TrueNode", "true"); }

void ASTPrinter::visit(FalseNode *node) { printTreeLine("FalseNode", "false"); }

void ASTPrinter::visit(CharNode *node) {
  printTreeLine("CharNode", "'" + std::string(1, node->value) + "'");
}

void ASTPrinter::visit(IntNode *node) {
  printTreeLine("IntNode", std::to_string(node->value));
}

void ASTPrinter::visit(RealNode *node) {
  printTreeLine("RealNode", std::to_string(node->value));
}

void ASTPrinter::visit(StringNode *node) {
  printTreeLine("StringNode", '"' + node->value + '"');
}

void ASTPrinter::visit(IdNode *node) { printTreeLine("IdNode", node->id); }

void ASTPrinter::visit(StructAccessNode *node) {
  printTreeLine("StructAccessNode", "name: " + node->structName + ", field: " + node->fieldName);
}

void ASTPrinter::visit(TypedDecNode *node) {
  printTreeLine("TypedDecNode", "name: " + node->name + ", qualifier: " +
                                    (node->qualifier.empty()
                                         ? "const"
                                         : node->qualifier));
  indent++;
  pushChildContext(node->init == nullptr);
  node->type_alias->accept(*this);
  popChildContext();

  if (node->init) {
    pushChildContext(true);
    indent++;
    pushChildContext(true);
    node->init->accept(*this);
    popChildContext();
    indent--;
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(InferredDecNode *node) {
  printTreeLine("InferredDecNode", "name: " + node->name + ", qualifier: " +
                                       (node->qualifier.empty()
                                            ? "const"
                                            : node->qualifier));
  indent++;
  pushChildContext(true);
  node->init->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(TupleTypedDecNode *node) {
  printTreeLine("TupleTypedDecNode", "name: " + node->name + ", qualifier: " +
                                      (node->qualifier.empty()
                                           ? std::string("const")
                                           : node->qualifier));
  indent++;
  pushChildContext(node->init == nullptr);
  printTreeLine("TupleType", toString(node->type));
  popChildContext();
  if (node->init) {
    pushChildContext(true);
    node->init->accept(*this);
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(StructTypedDecNode *node) {
  printTreeLine("StructTypedDecNode", "name: " + node->name + ", qualifier: " +
                                       (node->qualifier.empty()
                                            ? std::string("const")
                                            : node->qualifier));
  indent++;
  pushChildContext(node->init == nullptr);
  printTreeLine("StructType", toString(node->type));
  popChildContext();
  if (node->init) {
    pushChildContext(true);
    node->init->accept(*this);
    popChildContext();
  }
  indent--;
  }

void ASTPrinter::visit(AssignStatNode *node) {
  printTreeLine("AssignStatNode", "name: " + node->name);
  indent++;
  pushChildContext(true);
  node->expr->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(DestructAssignStatNode *node) {
  std::string targets;
  for (size_t i = 0; i < node->names.size(); ++i) {
    if (i > 0)
      targets += ", ";
    targets += node->names[i];
  }
  printTreeLine("DestructAssignStatNode", "targets: " + targets);
  indent++;
  pushChildContext(true);
  if (node->expr) {
    node->expr->accept(*this);
  } else {
    printTreeLine("ERROR: null expression");
  }
  popChildContext();
  indent--;
}

void ASTPrinter::visit(TupleAccessAssignStatNode *node) {
  std::string targetStr = node->target
                              ? node->target->tupleName + "." +
                                    std::to_string(node->target->index)
                              : "<null>";
  printTreeLine("TupleAccessAssignStatNode", "target: " + targetStr);
  indent++;
  pushChildContext(true);
  if (node->expr) {
    node->expr->accept(*this);
  } else {
    printTreeLine("ERROR: null expression");
  }
  popChildContext();
  indent--;
}

void ASTPrinter::visit(StructAccessAssignStatNode *node) {
  std::string targetStr = node->target ?
      node->target->structName + "." + node->target->fieldName
      : "<null>";

  printTreeLine("StructAccessAssignStatNode", "target: " + targetStr);
  indent++;
  pushChildContext(true);
  if (node->expr) {
    node->expr->accept(*this);
  } else {
    printTreeLine("ERROR: null expression");
  }
  popChildContext();
  indent--;
}

void ASTPrinter::visit(ArrayAccessAssignStatNode *node) {
  printTreeLine("ArrayAccessAssignStatNode");
  indent++;
  pushChildContext(false);
  printTreeLine("Target");
  indent++;
  pushChildContext(true);
  if (node->target) node->target->accept(*this);
  else printTreeLine("null target");
  popChildContext();
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("Expr");
  indent++;
  pushChildContext(true);
  if (node->expr) node->expr->accept(*this);
  else printTreeLine("null expr");
  popChildContext();
  indent--;
  popChildContext();
  indent--;
}

void ASTPrinter::visit(OutputStatNode *node) {
  if (!node) {
    printTreeLine("OutputStatNode", "ERROR: null node");
    return;
  }
  printTreeLine("OutputStatNode");
  indent++;
  pushChildContext(true);
  if (node->expr) {
    node->expr->accept(*this);
  } else {
    printTreeLine("ERROR: null expression");
  }
  popChildContext();
  indent--;
}

void ASTPrinter::visit(InputStatNode *node) {
  printTreeLine("InputStatNode", "name: " + node->name);
}

void ASTPrinter::visit(BreakStatNode *node) { printTreeLine("BreakStatNode"); }

void ASTPrinter::visit(ContinueStatNode *node) {
  printTreeLine("ContinueStatNode");
}

void ASTPrinter::visit(ReturnStatNode *node) {
  printTreeLine("ReturnStatNode");
  if (node->expr) {
    indent++;
    pushChildContext(true);
    node->expr->accept(*this);
    popChildContext();
    indent--;
  }
}

void ASTPrinter::visit(FuncCallExprOrStructLiteral *node) {
  printTreeLine("FuncCallExprOrStructLiteral", "name: " + node->funcName);
  indent++;
  for (size_t i = 0; i < node->args.size(); ++i) {
    pushChildContext(i == node->args.size() - 1);
    node->args[i]->accept(*this);
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(CallStatNode *node) {
  printTreeLine("CallStatNode");
  indent++;
  pushChildContext(true);
  node->call->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(FuncStatNode *node) {
  printTreeLine("FuncStatNode", "name: " + node->name);
  indent++;

  pushChildContext(false);
  printTreeLine("ReturnType", toString(node->returnType));
  popChildContext();

  pushChildContext(false);
  printTreeLine("Parameters");
  indent++;
  for (size_t i = 0; i < node->parameters.size(); ++i) {
    auto &p = node->parameters[i];
    pushChildContext(i == node->parameters.size() - 1);
    printTreeLine("Param",
                  p.identifier + ": " + toString(p.type));
    popChildContext();
  }
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("StatementBody");
  indent++;
  pushChildContext(true);
  node->returnStat->accept(*this);
  popChildContext();
  indent--;
  popChildContext();

  indent--;
}

void ASTPrinter::visit(FuncPrototypeNode *node) {
  printTreeLine("FuncPrototypeNode", "name: " + node->name);
  indent++;

  pushChildContext(false);
  printTreeLine("ReturnType", toString(node->returnType));
  popChildContext();

  pushChildContext(true);
  printTreeLine("Parameters");
  indent++;
  for (size_t i = 0; i < node->parameters.size(); ++i) {
    auto &p = node->parameters[i];
    pushChildContext(i == node->parameters.size() - 1);
    printTreeLine("Param",
                  p.identifier + ": " + toString(p.type));
    popChildContext();
  }
  indent--;
  popChildContext();
  indent--;
}

void ASTPrinter::visit(FuncBlockNode *node) {
  printTreeLine("FuncBlockNode", "name: " + node->name);
  indent++;

  pushChildContext(false);
  printTreeLine("ReturnType", toString(node->returnType));
  popChildContext();

  pushChildContext(false);
  printTreeLine("Parameters");
  indent++;
  for (size_t i = 0; i < node->parameters.size(); ++i) {
    auto &p = node->parameters[i];
    pushChildContext(i == node->parameters.size() - 1);
    printTreeLine("Param",
                  p.identifier + ": " + toString(p.type));
    popChildContext();
  }
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("Body");
  indent++;
  pushChildContext(true);
  node->body->accept(*this);
  popChildContext();
  indent--;
  popChildContext();

  indent--;
}

void ASTPrinter::visit(ProcedureBlockNode *node) {
  printTreeLine("ProcedureBlockNode", "name: " + node->name);
  indent++;

  pushChildContext(false);
  printTreeLine("Parameters");
  indent++;
  for (size_t i = 0; i < node->params.size(); ++i) {
    auto &p = node->params[i];
    pushChildContext(i == node->params.size() - 1);
    printTreeLine("Param",
                  p.identifier + ": " + toString(p.type));
    popChildContext();
  }
  indent--;
  popChildContext();

  pushChildContext(true);
  printTreeLine("Body");
  indent++;
  pushChildContext(true);
  node->body->accept(*this);
  popChildContext();
  indent--;
  popChildContext();

  indent--;
}

void ASTPrinter::visit(ProcedurePrototypeNode *node) {
  printTreeLine("ProcedurePrototypeNode", "name: " + node->name);
  indent++;

  pushChildContext(true);
  printTreeLine("Parameters");
  indent++;
  for (size_t i = 0; i < node->params.size(); ++i) {
    auto &p = node->params[i];
    pushChildContext(i == node->params.size() - 1);
    printTreeLine("Param",
                  p.identifier + ": " + toString(p.type));
    popChildContext();
  }
  indent--;
  popChildContext();

  indent--;
}

void ASTPrinter::visit(TypeAliasDecNode *node) {
  printTreeLine("TypeAliasDecNode",
                "alias: " + node->alias + " -> " + toString(node->type));
}

void ASTPrinter::visit(TypeAliasNode *node) {
  if (!node->aliasName.empty()) {
    printTreeLine("TypeAliasNode", "alias: " + node->aliasName);
  } else {
    printTreeLine("TypeAliasNode", "type: " + toString(node->type));
  }
}

void ASTPrinter::visit(TupleTypeAliasNode *node) {
  printTreeLine("TupleTypeAliasNode",
                "alias: " + node->aliasName + " -> " + toString(node->type));
}

void ASTPrinter::visit(TupleLiteralNode *node) {
  printTreeLine("TupleLiteralNode");
  indent++;
  for (size_t i = 0; i < node->elements.size(); i++) {
    pushChildContext(i == node->elements.size() - 1);
    node->elements[i]->accept(*this);
    popChildContext();
  }
  indent--;
}

void ASTPrinter::visit(TupleAccessNode *node) {
  if (!node) {
    printTreeLine("TupleAccessNode", "ERROR: null node");
    return;
  }
  printTreeLine("TupleAccessNode", "name: " + node->tupleName +
                                       ", index: " + std::to_string(node->index));
}

void ASTPrinter::visit(TypeCastNode *node) {
  printTreeLine("TypeCastNode", "target: " + toString(node->targetType));
  indent++;
  pushChildContext(true);
  node->expr->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::visit(TupleTypeCastNode *node) {
  printTreeLine("TupleTypeCastNode", "target: " + toString(node->targetTupleType));
  indent++;
  pushChildContext(true);
  node->expr->accept(*this);
  popChildContext();
  indent--;
}

void ASTPrinter::printTreeLine(const std::string &nodeType,
                               const std::string &details) {
  for (int i = 0; i < indent; i++) {
    if (i == 0) {
      out << "    ";
      continue;
    }
    bool hasMoreSiblingsAtAncestor =
        !isLastChild[static_cast<size_t>(i - 1)];
    if (hasMoreSiblingsAtAncestor) {
      out << (colorEnabled ? "│   " : "|   ");
    } else {
      out << "    ";
    }
  }

  if (indent > 0) {
    if (!isLastChild.empty() && isLastChild.back()) {
      out << (colorEnabled ? "└── " : "`-- ");
    } else {
      out << (colorEnabled ? "├── " : "|-- ");
    }
  }

  out << nodeType;
  if (!details.empty()) {
    out << " (" << details << ")";
  }
  out << '\n';
}

void ASTPrinter::pushChildContext(bool isLast) {
  isLastChild.push_back(isLast);
}

void ASTPrinter::popChildContext() {
  if (!isLastChild.empty()) {
    isLastChild.pop_back();
  }
}

} // namespace gazprea
