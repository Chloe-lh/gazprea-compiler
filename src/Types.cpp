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

ValueType promote(ValueType from, ValueType to)
{
    switch (from) {
        case ValueType::UNKNOWN:
            return ValueType::UNKNOWN;
            break;
        case ValueType::BOOL:
            switch(to) {
                case ValueType::BOOL:       return ValueType::BOOL;

                // Implicit promotions
                case ValueType::ARRAY:      return ValueType::ARRAY;
                case ValueType::VECTOR:     return ValueType::VECTOR; 
                case ValueType::MATRIX:     return ValueType::MATRIX;
            }
            break;
        case ValueType::INTEGER:
            switch (to) {
                case ValueType::INTEGER:    return ValueType::INTEGER; 
                case ValueType::REAL:       return ValueType::REAL;

                // Implicit promotions
                case ValueType::ARRAY:      return ValueType::ARRAY;
                case ValueType::VECTOR:     return ValueType::VECTOR; 
                case ValueType::MATRIX:     return ValueType::MATRIX;
            }
            break;
        case ValueType::VECTOR:
            switch (to) {
                case ValueType::VECTOR:     return ValueType::VECTOR;
            }
            break;
        case ValueType::REAL:
            switch (to) {
                case ValueType::REAL:       return ValueType::REAL;

                // Implicit promotions
                case ValueType::ARRAY:      return ValueType::ARRAY;
                case ValueType::VECTOR:     return ValueType::VECTOR; 
                case ValueType::MATRIX:     return ValueType::MATRIX;
            }
            break;
        case ValueType::TUPLE:
            switch (to) {
                case ValueType::TUPLE:      return ValueType::TUPLE;    // only if same len + internal types are convertible element-wise
            }
            break;
    }
    return ValueType::UNKNOWN;
}