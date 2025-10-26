#pragma once

#include <string>
#include <vector>
#include <variant>

// Type enum for all nodes
enum class ValueType {
    UNKNOWN,
    INTEGER,
    VECTOR
};

std::string toString(ValueType type);

inline ValueType promote(ValueType from, ValueType to) {
    switch (from) {
        case ValueType::UNKNOWN:
            return ValueType::UNKNOWN;
            break;
        case ValueType::INTEGER:
            switch (to) {
                case ValueType::UNKNOWN: return ValueType::UNKNOWN;
                case ValueType::INTEGER: return ValueType::INTEGER;
                case ValueType::VECTOR:  return ValueType::VECTOR;
            }
            break;
        case ValueType::VECTOR:
            switch (to) {
                case ValueType::UNKNOWN: return ValueType::UNKNOWN;
                case ValueType::INTEGER: return ValueType::UNKNOWN;
                case ValueType::VECTOR:  return ValueType::VECTOR;
            }
            break;
    }
    return ValueType::UNKNOWN;
}
