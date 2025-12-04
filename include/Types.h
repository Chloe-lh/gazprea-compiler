#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <assert.h>

// Base Types
enum class BaseType {
    UNRESOLVED,
    UNKNOWN,
    EMPTY,
    BOOL,
    CHARACTER,
    INTEGER,
    REAL,
    TUPLE,
    VECTOR,
    STRING,
    STRUCT,
    MATRIX,
    ARRAY
};

inline constexpr BaseType flatTypes[] = {
    BaseType::BOOL,
    BaseType::CHARACTER,
    BaseType::INTEGER,
    BaseType::REAL
};

inline constexpr BaseType compositeTypes[] = {
    BaseType::TUPLE,
    BaseType::VECTOR,
    BaseType::STRING,
    BaseType::STRUCT,
    BaseType::MATRIX,
    BaseType::ARRAY
};

struct CompleteType {
    BaseType baseType;
    std::vector<CompleteType> subTypes; // used for composite types (tuple, struct, array, vector, matrix)
    std::string aliasName = "";
    std::vector<std::string> fieldNames; //stores field names for structs only
    std::vector<int> dims; // Dimension metadata for arrays / vectors / matrices.

    CompleteType(): baseType(BaseType::UNKNOWN) {}
    CompleteType(BaseType baseType) : baseType(baseType) {}
    CompleteType(BaseType baseType, std::vector<CompleteType> subTypes)
        : baseType(baseType), subTypes(std::move(subTypes)) {}
    CompleteType(std::string aliasName) : baseType(BaseType::UNRESOLVED), aliasName(aliasName) {} // constructor for type aliases - actual type resolved during semantic analysis
    CompleteType(BaseType baseType, CompleteType subType, std::vector<int> dims)
        : baseType(baseType), subTypes({subType}), dims(std::move(dims)) {} // constructor for arrays

    bool operator==(const CompleteType& other) const noexcept {
        return baseType == other.baseType &&
               subTypes == other.subTypes &&
               dims == other.dims;
    }
    bool operator!=(const CompleteType& other) const noexcept {
        return !(*this == other);
    }
};

// Stringify base types (primitive kind only)
std::string toString(BaseType type);

// Stringify complete types, including any subtype information
// For example: tuple(integer, real), vector(integer)
std::string toString(const CompleteType& type);

BaseType promote(BaseType from, BaseType to);

// Promote a full type, including nested subtypes. Returns UNKNOWN on failure.
CompleteType promote(const CompleteType& from, const CompleteType& to);

void validateSubtypes(CompleteType completeType);

// Type casting helpers (semantic layer consumes these)
// Returns true iff the BaseType is a scalar (boolean, character, integer, real)
bool isScalarType(BaseType t);

// Returns true iff a scalar of 'from' type can be explicitly cast to 'to' type
// according to the Scalar to Scalar casting rules in the language spec.
bool canScalarCast(BaseType from, BaseType to);

// Returns true iff a value of CompleteType 'from' can be explicitly cast to
// CompleteType 'to' according to the casting rules in the language spec.
// Implemented for:
//  - scalar -> scalar (full per spec)
//  - tuple  -> tuple  (pairwise scalar-castable, equal arity)
//  - identity for identical shapes (recursively equal base + compatible subtypes)
//
// TODO pt2: Extend to support scalar<->array promotions and array<->array/matrix/vector
//           conversions once size/dimension information is modeled in CompleteType.
bool canCastType(const CompleteType& from, const CompleteType& to);
