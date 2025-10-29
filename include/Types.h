#pragma once

#include <string>
#include <vector>
#include <variant>
#include <assert.h>

// Base Types
enum class BaseType {
    UNKNOWN,
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

BaseType flatTypes[] = {
    BaseType::BOOL,
    BaseType::CHARACTER,
    BaseType::INTEGER,
    BaseType::REAL
};

BaseType compositionTypes[] = {
    BaseType::TUPLE,
    BaseType::VECTOR,
    BaseType::STRING,
    BaseType::STRUCT,
    BaseType::MATRIX,
    BaseType::ARRAY
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

// Promote a full type, including nested subtypes. Returns UNKNOWN on failure.
CompleteType promote(const CompleteType& from, const CompleteType& to);

void validateSubtypes(CompleteType completeType);
