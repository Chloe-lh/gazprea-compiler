#pragma once

#include <string>
#include <vector>
#include <variant>

// Type enum for all nodes
enum class ValueType {
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

std::string toString(ValueType type);

ValueType promote(ValueType from, ValueType to);
