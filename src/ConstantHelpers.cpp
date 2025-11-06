#include "ConstantHelpers.h"
#include "AST.h"
#include "Types.h"
#include <cmath>
#include <optional>
#include <cassert>

namespace gazprea {

// Compute a numeric binary operation between two constants. Supported ops:
// "+", "-", "*", "/", "%", "^". If operands promote to real
// (double), the result will be a real ConstantValue. For integer-only
// operands, result is integer (int64_t) for +,-,*,% and REAL for / and ^.
// Returns std::nullopt on type mismatch or divide-by-zero.
std::optional<ConstantValue>
computeBinaryNumeric(const ConstantValue &a, const ConstantValue &b,
                     const std::string &op) {
  // Try real operands first (or promotion)
  auto aReal = tryGetRealOpt(a);
  auto bReal = tryGetRealOpt(b);
  if (aReal && bReal) {
    double av = *aReal;
    double bv = *bReal;
    if (op == "+") return ConstantValue(CompleteType(BaseType::REAL), av + bv);
    if (op == "-") return ConstantValue(CompleteType(BaseType::REAL), av - bv);
    if (op == "*") return ConstantValue(CompleteType(BaseType::REAL), av * bv);
    if (op == "/") {
      if (bv == 0.0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::REAL), av / bv);
    }
    if (op == "^") return ConstantValue(CompleteType(BaseType::REAL), std::pow(av, bv));
    if (op == "%") {
      if (bv == 0.0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::REAL), std::fmod(av, bv));
    }
    return std::nullopt;
  }

  // Try integer-int case
  auto aInt = tryGetIntOpt(a);
  auto bInt = tryGetIntOpt(b);
  if (aInt && bInt) {
    int64_t av = *aInt;
    int64_t bv = *bInt;
    if (op == "+") return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av + bv));
    if (op == "-") return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av - bv));
    if (op == "*") return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av * bv));
    if (op == "%") {
      if (bv == 0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av % bv));
    }
    if (op == "/") {
      if (bv == 0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::REAL), static_cast<double>(av) / static_cast<double>(bv));
    }
    if (op == "^") return ConstantValue(CompleteType(BaseType::REAL), std::pow(static_cast<double>(av), static_cast<double>(bv)));
    return std::nullopt;
  }

  // Mixed int/real -> promote to real and compute
  if (aReal || bReal) {
    // we can safely obtain int options here (may be nullopt)
    auto aInt2 = tryGetIntOpt(a);
    auto bInt2 = tryGetIntOpt(b);
    double aval = aReal ? *aReal : static_cast<double>(*aInt2);
    double bval = bReal ? *bReal : static_cast<double>(*bInt2);
    if (op == "+") return ConstantValue(CompleteType(BaseType::REAL), aval + bval);
    if (op == "-") return ConstantValue(CompleteType(BaseType::REAL), aval - bval);
    if (op == "*") return ConstantValue(CompleteType(BaseType::REAL), aval * bval);
    if (op == "/") {
      if (bval == 0.0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::REAL), aval / bval);
    }
    if (op == "%") {
      if (bval == 0.0) return std::nullopt;
      return ConstantValue(CompleteType(BaseType::REAL), std::fmod(aval, bval));
    }
    if (op == "^") return ConstantValue(CompleteType(BaseType::REAL), std::pow(aval, bval));
    return std::nullopt;
  }

  return std::nullopt;
}

// ++a, --a, not(a)
std::optional<ConstantValue> computeUnaryNumeric(const ConstantValue &a, const std::string &op) {
  auto aReal = tryGetRealOpt(a);
  int delta = 1;
  if (aReal) {
    double av = *aReal;
    if (op == "++") return ConstantValue(CompleteType(BaseType::REAL), av + delta);
    if (op == "--") return ConstantValue(CompleteType(BaseType::REAL), av - delta);
    if (op == "not") return ConstantValue(CompleteType(BaseType::BOOL), av == 0.0 ? false : true);
  }

  auto aInt = tryGetIntOpt(a);
  if (aInt) {
    int64_t av = *aInt;
    if (op == "++") return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av + delta));
    if (op == "--") return ConstantValue(CompleteType(BaseType::INTEGER), static_cast<int64_t>(av - delta));
    if (op == "not") return ConstantValue(CompleteType(BaseType::BOOL), av == 0 ? false : true);
  }
  return std::nullopt;
}

std::optional<ConstantValue> computeBinaryComp(const ConstantValue &a, const ConstantValue &b, const std::string &op) {
  // Try integer comparison first
  auto aInt = tryGetIntOpt(a);
  auto bInt = tryGetIntOpt(b);
  if (aInt && bInt) {
    int64_t av = *aInt;
    int64_t bv = *bInt;
    if (op == "and") return ConstantValue(CompleteType(BaseType::BOOL), (av != 0 && bv != 0));
    if (op == "or") return ConstantValue(CompleteType(BaseType::BOOL), (av != 0 || bv != 0));
    if (op == "xor") return ConstantValue(CompleteType(BaseType::BOOL), ((av != 0) != (bv != 0)));
    if (op == "==") return ConstantValue(CompleteType(BaseType::BOOL), (av == bv));
    if (op == "!=") return ConstantValue(CompleteType(BaseType::BOOL), (av != bv));
    if (op == ">") return ConstantValue(CompleteType(BaseType::BOOL), (av > bv));
    if (op == "<") return ConstantValue(CompleteType(BaseType::BOOL), (av < bv));
    if (op == ">=") return ConstantValue(CompleteType(BaseType::BOOL), (av >= bv));
    if (op == "<=") return ConstantValue(CompleteType(BaseType::BOOL), (av <= bv));
  }

  // If either operand is real (or both), perform real comparisons (promote ints)
  auto aReal = tryGetRealOpt(a);
  auto bReal = tryGetRealOpt(b);
  if (aReal || bReal) {
    // promote ints if needed
    auto aInt2 = tryGetIntOpt(a);
    auto bInt2 = tryGetIntOpt(b);
    double aval = aReal ? *aReal : static_cast<double>(*aInt2);
    double bval = bReal ? *bReal : static_cast<double>(*bInt2);
    if (op == "and") return ConstantValue(CompleteType(BaseType::BOOL), (aval != 0.0 && bval != 0.0));
    if (op == "or") return ConstantValue(CompleteType(BaseType::BOOL), (aval != 0.0 || bval != 0.0));
    if (op == "xor") return ConstantValue(CompleteType(BaseType::BOOL), ((aval != 0.0) != (bval != 0.0)));
    if (op == "==") return ConstantValue(CompleteType(BaseType::BOOL), (aval == bval));
    if (op == "!=") return ConstantValue(CompleteType(BaseType::BOOL), (aval != bval));
    if (op == ">") return ConstantValue(CompleteType(BaseType::BOOL), (aval > bval));
    if (op == "<") return ConstantValue(CompleteType(BaseType::BOOL), (aval < bval));
    if (op == ">=") return ConstantValue(CompleteType(BaseType::BOOL), (aval >= bval));
    if (op == "<=") return ConstantValue(CompleteType(BaseType::BOOL), (aval <= bval));
  }

  return std::nullopt;
}

} // namespace gazprea