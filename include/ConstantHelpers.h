// Helpers for working with ConstantValue (promotion and numeric ops)
#pragma once
#include "AST.h"
#include <optional>
#include <string>

namespace gazprea {

// Compute a numeric binary operation between two constants. Supported ops:
// "+", "-", "*", "/", "%". If operands promote to real (double), the
// result will be a real ConstantValue. For integer-only operands, result is
// integer (int64_t). Returns std::nullopt on type mismatch or divide-by-zero.
std::optional<ConstantValue> computeBinaryNumeric(const ConstantValue &a,
                                                 const ConstantValue &b,
                                                 const std::string &op);
std::optional<ConstantValue> computeUnaryNumeric(const ConstantValue &a, const std::string &op);
std::optional<ConstantValue> computeBinaryComp(const ConstantValue &a, const ConstantValue &b, const std::string &op);



// returns present int64_t or nullopt
static inline std::optional<int64_t> tryGetIntOpt(const ConstantValue &cv) {
  if (auto p = std::get_if<int64_t>(&cv.value)) return *p;
  return std::nullopt;
}

// returns a promoted double: uses double if present, else promotes int64_t to double
static inline std::optional<double> tryGetRealOpt(const ConstantValue &cv) {
  if (auto pd = std::get_if<double>(&cv.value)) return *pd;
  if (auto pi = std::get_if<int64_t>(&cv.value)) return static_cast<double>(*pi);
  return std::nullopt;
}


} // namespace gazprea
