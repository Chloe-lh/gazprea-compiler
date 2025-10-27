#include "Types.h"

std::string toString(ValueType type) {
    switch (type) {
        case ValueType::UNKNOWN:   return "unknown";
        case ValueType::BOOL:      return "boolean";
        case ValueType::CHARACTER: return "character";
        case ValueType::INTEGER:   return "integer";
        case ValueType::VECTOR:    return "vector";
        case ValueType::REAL:      return "real";
        case ValueType::TUPLE:     return "tuple";
        case ValueType::STRING:    return "string";
        case ValueType::STRUCT:    return "struct";
    }
    return "unknown";
}