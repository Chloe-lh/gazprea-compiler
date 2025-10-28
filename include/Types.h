#pragma once

#include <string>
#include <vector>
#include <variant>

// Base Types
enum class BaseType {
    UNKNOWN,
    BOOL,
    CHARACTER,
    INTEGER,
    VECTOR,
    REAL,
    TUPLE,
    STRING,
    STRUCT,
    MATRIX,
    ARRAY
};

struct CompleteType {
    BaseType baseType;
    std::vector<CompleteType> subTypes;

    CompleteType(BaseType baseType) : baseType(baseType) {}
    CompleteType(BaseType baseType, std::vector<CompleteType> subTypes)
        : baseType(baseType), subTypes(std::move(subTypes)) {}
};

// Stringify base types (primitive kind only)
std::string toString(BaseType type);

// Stringify complete types, including any subtype information
// For example: tuple(integer, real), vector(integer)
std::string toString(const CompleteType& type);

BaseType promote(BaseType from, BaseType to);
