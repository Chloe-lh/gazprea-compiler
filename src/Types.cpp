#include "Types.h"

std::string toString(ValueType type) {
    switch (type) {
        case ValueType::UNKNOWN:    return "unknown";
        case ValueType::BOOL:       return "boolean";
        case ValueType::CHARACTER:  return "character";
        case ValueType::INTEGER:    return "integer";
        case ValueType::VECTOR:     return "vector";
        case ValueType::REAL:       return "real";
        case ValueType::TUPLE:      return "tuple";
        case ValueType::STRING:     return "string";
        case ValueType::STRUCT:     return "struct";
        case ValueType::ARRAY:      return "array";
        case ValueType::MATRIX:     return "matrix";
    }
    return "unknown";
}
/* TODO pt2
    - handle character array <-> character vector <-> string promotions
*/
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
        case ValueType::CHARACTER:
            switch (to) {
                case ValueType::CHARACTER:  return ValueType::CHARACTER; 
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
                case ValueType::TUPLE:      return ValueType::TUPLE;    // only valid if same len + internal types are convertible element-wise
            }
            break;
        case ValueType::STRING:
            switch (to) {
                case ValueType::STRING:     return ValueType::STRING;   // not the same as vector<char>. 
            }
            break;
        case ValueType::STRUCT:
            switch (to) {
                case ValueType::STRUCT:     return ValueType::STRUCT;    
            }
            break;
        case ValueType::ARRAY:
            switch (to) {
                case ValueType::ARRAY:      return ValueType::ARRAY;    // only valid if same len
            }
            break;
        case ValueType::MATRIX:
            switch (to) {
                case ValueType::MATRIX:     return ValueType::MATRIX;   // only valid if same len - special case with multiplication.
            }
            break;
    }
    return ValueType::UNKNOWN;
}
